"""Maison Cognitive mc10 runtime refinements.

mc10 builds on the validated mc9 behaviour and changes only two things:

1. Prevent duplicate/intermediate HomePod replies around tool calls. Tool-like
   fillers ("je vérifie", "une seconde", etc.) keep a stronger 1.5 s guard,
   while normal complete replies keep a short 0.25 s guard. Incomplete text
   still uses the proven 2.0 s continuation window.
2. Measure where HomePod/AirPlay startup time is spent. Morgan URL generation
   and HomePod media-state transition are timed separately while play_media is
   running; play_media itself is still awaited so follow-up only reopens after
   the HomePod stream has really ended.

mc9 is imported explicitly here so its immediate post-HomePod follow-up and
fragment-merging logic stay intact.
"""

import asyncio
import json
import logging
import time
import unicodedata
import urllib.error
import urllib.request

import app.mc9_patch  # installs the mc9 baseline first
from app.homepod_speech_router import HomePodSpeechRouter

logger = logging.getLogger("app.homepod_speech_router")

_MC9_ROUTER_INIT = HomePodSpeechRouter.__init__

_TOOL_FILLER_STEMS = (
    "je verif",
    "je vais verif",
    "je regard",
    "je vais regard",
    "je consult",
    "je cherch",
    "je recup",
    "verifions",
    "laisse-moi",
    "laissez-moi",
    "attend",
    "checking",
    "let me check",
    "let me look",
)

_TOOL_FILLER_ANYWHERE = (
    "une seconde",
    "un instant",
    "one moment",
)


def _normalize(text: str) -> str:
    text = " ".join((text or "").strip().lower().split())
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _looks_like_tool_filler(text: str) -> bool:
    clean = _normalize(text)
    return (
        any(clean.startswith(stem) for stem in _TOOL_FILLER_STEMS)
        or any(token in clean for token in _TOOL_FILLER_ANYWHERE)
    )


def _mc10_router_init(self, *args, **kwargs):
    _MC9_ROUTER_INIT(self, *args, **kwargs)
    self._mc10_complete_hold_seconds = 0.25
    self._mc10_tool_filler_hold_seconds = 1.5
    self._continuation_hold_seconds = 2.0
    self._mc10_raop_probe_seconds = 8.0
    self._mc10_raop_probe_interval = 0.12


def _mc10_hold_seconds_for_text(self, text: str, *, post_tool: bool) -> float:
    if self._looks_incomplete(text) and len(text) <= self._continuation_hold_max_chars:
        return self._continuation_hold_seconds

    if post_tool:
        return 0.0

    if len(text) <= self._pretool_hold_max_chars:
        if _looks_like_tool_filler(text):
            return self._mc10_tool_filler_hold_seconds
        return self._mc10_complete_hold_seconds

    return 0.0


def _state_marker(data: dict | None):
    if not isinstance(data, dict):
        return None
    attrs = data.get("attributes") or {}
    return (
        data.get("state"),
        attrs.get("media_content_id"),
        attrs.get("media_title"),
        attrs.get("media_position_updated_at"),
        attrs.get("media_duration"),
        attrs.get("app_name"),
    )


def _get_homepod_state_sync(router) -> dict:
    token = router._get_ha_token()
    if not token:
        return {}
    request = urllib.request.Request(
        url=f"{router._ha_api_base}/states/{router._target_entity}",
        method="GET",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            payload = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return {}
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


async def _probe_raop_start(router, before: dict, play_started_at: float, play_task):
    before_marker = _state_marker(before)
    before_state = before.get("state") if isinstance(before, dict) else None

    while (time.monotonic() - play_started_at) < router._mc10_raop_probe_seconds:
        await asyncio.sleep(router._mc10_raop_probe_interval)
        current = await asyncio.to_thread(_get_homepod_state_sync, router)
        current_marker = _state_marker(current)
        current_state = current.get("state") if isinstance(current, dict) else None

        changed = bool(current_marker and current_marker != before_marker)
        started_playing = current_state == "playing" and before_state != "playing"
        if changed or started_playing:
            elapsed = time.monotonic() - play_started_at
            logger.info(
                "⏱️ HomePod router: transition RAOP détectée après %.2fs "
                "(état %s -> %s)",
                elapsed,
                before_state,
                current_state,
            )
            return elapsed

        if play_task.done():
            return None

    return None


async def _mc10_speak_on_homepod(self, text: str) -> tuple[bool, float]:
    started = time.monotonic()
    token = self._get_ha_token()
    if not token:
        logger.warning("⚠️ HomePod router: aucun jeton HA, fallback Voice PE")
        return False, 0.0

    try:
        # 1) Generate the lazy Morgan stream URL.
        url_started = time.monotonic()
        tts_data = await asyncio.to_thread(
            self._request_json_sync,
            f"{self._ha_api_base}/tts_get_url",
            {
                "engine_id": self._tts_entity,
                "message": text,
                "cache": False,
            },
            token,
        )
        url_elapsed = time.monotonic() - url_started
        media_url = str(tts_data.get("url") or "").strip()
        if not media_url:
            raise RuntimeError("/api/tts_get_url n'a renvoyé aucune URL")

        logger.info(
            "🐟 HomePod router: URL Morgan prête en %.2fs via %s",
            url_elapsed,
            self._tts_entity,
        )

        # 2) Snapshot HomePod state, then start play_media in one worker while a
        # second worker polls HA state. This gives us an approximation of the
        # AirPlay/RAOP start transition instead of confusing full playback time
        # with startup latency.
        before = await asyncio.to_thread(_get_homepod_state_sync, self)
        play_started_at = time.monotonic()
        play_task = asyncio.create_task(
            asyncio.to_thread(
                self._request_service_sync,
                "media_player",
                "play_media",
                {
                    "entity_id": self._target_entity,
                    "media_content_id": media_url,
                    "media_content_type": "music",
                    "announce": True,
                },
                token,
            )
        )
        probe_task = asyncio.create_task(
            _probe_raop_start(self, before, play_started_at, play_task)
        )

        try:
            await play_task
        finally:
            if not probe_task.done():
                probe_task.cancel()
                try:
                    await probe_task
                except asyncio.CancelledError:
                    pass

        play_elapsed = time.monotonic() - play_started_at
        total_elapsed = time.monotonic() - started

        if probe_task.done() and not probe_task.cancelled():
            try:
                transition = probe_task.result()
            except Exception:
                transition = None
            if transition is None:
                logger.info(
                    "⏱️ HomePod router: aucune transition d'état RAOP exploitable "
                    "avant la fin de play_media"
                )

        logger.info(
            "🍎 HomePod router: play_media terminé en %.2fs vers %s",
            play_elapsed,
            self._target_entity,
        )
        logger.info(
            "⚡ HomePod router: fast path Morgan accepté "
            "(%d caractères, URL %.2fs, play_media %.2fs, total %.2fs)",
            len(text),
            url_elapsed,
            play_elapsed,
            total_elapsed,
        )
        return True, total_elapsed

    except Exception as fast_exc:
        logger.warning(
            "↩️ HomePod router: fast path mc10 indisponible, fallback tts.speak: %r",
            fast_exc,
        )

    try:
        fallback_started = time.monotonic()
        await asyncio.to_thread(self._call_home_assistant_legacy_sync, text)
        fallback_elapsed = time.monotonic() - fallback_started
        total_elapsed = time.monotonic() - started
        logger.info(
            "🏠 HomePod router: Morgan TTS accepté par fallback tts.speak "
            "(%d caractères, fallback %.2fs, total %.2fs)",
            len(text),
            fallback_elapsed,
            total_elapsed,
        )
        return True, total_elapsed
    except Exception as exc:
        total_elapsed = time.monotonic() - started
        logger.warning(
            "⚠️ HomePod router: échec Morgan TTS, fallback Voice PE après %.2fs: %r",
            total_elapsed,
            exc,
        )
        return False, total_elapsed


HomePodSpeechRouter.__init__ = _mc10_router_init
HomePodSpeechRouter._hold_seconds_for_text = _mc10_hold_seconds_for_text
HomePodSpeechRouter._speak_on_homepod = _mc10_speak_on_homepod

logger.info(
    "🚀 Maison Cognitive mc10 chargé: fillers 1.5s, réponses complètes 0.25s, "
    "continuation 2.0s, diagnostic démarrage RAOP actif"
)
