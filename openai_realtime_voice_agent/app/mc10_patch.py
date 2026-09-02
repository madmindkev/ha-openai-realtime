"""Maison Cognitive mc10 runtime refinements.

mc10 builds on the validated mc9 behaviour and changes only two things:

1. Prevent duplicate/intermediate HomePod replies around tool calls. Tool-like
   fillers ("je vérifie", "une seconde", etc.) keep a stronger 1.5 s guard,
   while normal complete replies keep a short 0.25 s guard. Incomplete text
   still uses the proven 2.0 s continuation window.
2. Measure where HomePod/AirPlay startup time is spent. Morgan URL generation
   and HomePod media-state transition are timed separately while play_media is
   running. Once the current request is confirmed as playing, the router can
   return early while keeping a per-HomePod playback lock until the HTTP call
   really ends. This keeps follow-up responsive without overlapping RAOP
   streams.

mc9 is imported explicitly here so its immediate post-HomePod follow-up and
fragment-merging logic stay intact.
"""

import asyncio
import json
import logging
import os
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
    self._mc10_raop_probe_interval = 0.08
    self._mc10_early_return_on_raop = _mc10_env_bool(
        "HOMEPOD_EARLY_RETURN_ON_RAOP", True
    )
    # Home Assistant's Apple TV service keeps the HTTP request open until the
    # whole AirPlay file has finished.  Some versions do not expose a useful
    # playing/media_content_id transition while that request is open, so use a
    # conservative grace period as a second, bounded hand-off signal.
    self._mc10_early_return_grace_seconds = _mc10_env_float(
        "HOMEPOD_EARLY_RETURN_GRACE_SECONDS", 2.5, minimum=0.5
    )
    # pyatv permits only one RAOP stream per HomePod. Keep the HTTP
    # play_media call serialized even when the foreground coroutine returns
    # as soon as the current stream is confirmed playing.
    self._mc10_playback_lock = asyncio.Lock()
    # Tolérance de comparaison du volume : en deçà, volume_set est sauté.
    self._mc10_volume_tolerance = _mc10_env_float(
        "HOMEPOD_VOLUME_TOLERANCE", 0.02, minimum=0.0
    )
    # Le réglage de volume est opportuniste : il ne doit jamais retarder ni
    # rendre silencieuse une réponse Morgan si pyatv ne répond pas.
    self._mc10_volume_wait_seconds = _mc10_env_float(
        "HOMEPOD_VOLUME_WAIT_SECONDS", 0.20, minimum=0.0
    )


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


def _mc10_env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _mc10_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip().lower() in {"", "null", "none"}:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _conversation_volume_for_hour(hour: int) -> float:
    """Politique maison : jour 75 %, soirée 60 %, nuit 25 %."""
    return 0.25 if hour >= 23 or hour < 7 else 0.60 if hour >= 18 else 0.75


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


async def _probe_raop_start(
    router,
    before: dict,
    play_started_at: float,
    play_task,
    expected_media_url: str,
):
    before_marker = _state_marker(before)
    before_state = before.get("state") if isinstance(before, dict) else None

    while (time.monotonic() - play_started_at) < router._mc10_raop_probe_seconds:
        await asyncio.sleep(router._mc10_raop_probe_interval)
        current = await asyncio.to_thread(_get_homepod_state_sync, router)
        current_marker = _state_marker(current)
        current_state = current.get("state") if isinstance(current, dict) else None
        current_media_id = str(
            (current.get("attributes") or {}).get("media_content_id") or ""
        ).strip()

        changed = bool(current_marker and current_marker != before_marker)
        started_playing = current_state == "playing" and before_state != "playing"
        # An unrelated announcement can change this entity while our request is
        # blocked. Only confirm the transition when it belongs to this exact
        # tts_get_url result.
        media_matches = bool(
            expected_media_url and current_media_id == expected_media_url
        )
        if media_matches and current_state == "playing" and (changed or started_playing):
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


def _log_detached_volume(task: asyncio.Task) -> None:
    """Consume a late volume_set result so pyatv errors stay observable."""
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:
        logger.warning("⚠️ HomePod router: volume_set différé échoué: %r", exc)


def _log_detached_play(task: asyncio.Task) -> None:
    """Consume a background play_media result after an early RAOP return."""
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:
        logger.warning(
            "⚠️ HomePod router: play_media détaché échoué après démarrage confirmé: %r",
            exc,
        )


async def _mc10_speak_on_homepod(self, text: str) -> tuple[bool, float]:
    started = time.monotonic()
    token = self._get_ha_token()
    if not token:
        logger.warning("⚠️ HomePod router: aucun jeton HA, fallback Voice PE")
        return False, 0.0

    lock_started = time.monotonic()
    await self._mc10_playback_lock.acquire()
    lock_wait = time.monotonic() - lock_started
    lock_released = False
    detached_play = False

    def release_playback_lock(_task=None):
        nonlocal lock_released
        if lock_released:
            return
        lock_released = True
        self._mc10_playback_lock.release()

    try:
        try:
            # Un seul GET d'état sert au volume ET de ligne de base à la sonde RAOP :
            # une requête HTTP de moins sur le chemin critique.
            state_started = time.monotonic()
            before = await asyncio.to_thread(_get_homepod_state_sync, self)
            state_elapsed = time.monotonic() - state_started

            conversation_volume = _conversation_volume_for_hour(time.localtime().tm_hour)
            current_volume = None
            if isinstance(before, dict):
                raw_volume = (before.get("attributes") or {}).get("volume_level")
                if isinstance(raw_volume, (int, float)):
                    current_volume = float(raw_volume)

            volume_task = None
            volume_skipped = (
                current_volume is not None
                and abs(current_volume - conversation_volume) <= self._mc10_volume_tolerance
            )
            if volume_skipped:
                logger.info(
                    "🔈 HomePod router: volume déjà à %.2f, volume_set sauté",
                    conversation_volume,
                )
            else:
                # Lancé en parallèle de tts_get_url : volume_set n'est attendu qu'avant
                # play_media, seul instant où il doit réellement avoir pris effet.
                volume_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._request_service_sync,
                        "media_player",
                        "volume_set",
                        {
                            "entity_id": self._target_entity,
                            "volume_level": conversation_volume,
                        },
                        token,
                    )
                )

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

            # 2) Start play_media in one worker while a second worker polls HA.
            # The lock stays held by the background task if we return early.
            volume_elapsed = 0.0
            if volume_task is not None:
                volume_started = time.monotonic()
                try:
                    # La commande peut rester bloquée ~5 s dans pyatv/MRP. Elle ne
                    # doit pas empêcher tts_get_url ni play_media de démarrer.
                    done, _ = await asyncio.wait(
                        {volume_task}, timeout=self._mc10_volume_wait_seconds
                    )
                    if volume_task not in done:
                        raise asyncio.TimeoutError
                    volume_task.result()
                    volume_elapsed = time.monotonic() - volume_started
                except asyncio.TimeoutError:
                    volume_elapsed = time.monotonic() - volume_started
                    volume_task.add_done_callback(_log_detached_volume)
                    logger.warning(
                        "⚠️ HomePod router: volume_set encore en cours après %.2fs; "
                        "poursuite de la réponse Morgan",
                        volume_elapsed,
                    )
                except Exception as volume_exc:
                    volume_elapsed = time.monotonic() - volume_started
                    logger.warning(
                        "⚠️ HomePod router: volume_set ignoré après %.2fs: %r",
                        volume_elapsed,
                        volume_exc,
                    )

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
                _probe_raop_start(self, before, play_started_at, play_task, media_url)
            )

            try:
                if self._mc10_early_return_on_raop:
                    done, _ = await asyncio.wait(
                        {play_task, probe_task},
                        timeout=self._mc10_early_return_grace_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if probe_task in done and not play_task.done():
                        try:
                            transition = probe_task.result()
                        except Exception:
                            transition = None
                        if transition is not None:
                            detached_play = True
                            play_task.add_done_callback(_log_detached_play)
                            play_task.add_done_callback(release_playback_lock)
                            total_elapsed = time.monotonic() - started
                            logger.info(
                                "⏱️ HomePod router: retour rapide après démarrage RAOP "
                                "%.2fs (attente verrou %.2fs, total %.2fs, play_media "
                                "maintenu en arrière-plan; %d car.)",
                                transition,
                                lock_wait,
                                total_elapsed,
                                len(text),
                            )
                            return True, total_elapsed

                    # The Apple TV integration can leave the entity state at
                    # idle until the complete play_media call returns.  Do not
                    # keep the conversation blocked for the whole audio file:
                    # once the request has remained alive for the bounded grace
                    # period, hand the foreground back and keep the HTTP call
                    # serialized by the playback lock.  If the task completed
                    # (successfully or with an error), the normal path below
                    # still observes its real result and can use the fallback.
                    if not play_task.done():
                        if not probe_task.done():
                            probe_task.cancel()
                            try:
                                await probe_task
                            except asyncio.CancelledError:
                                pass
                        detached_play = True
                        play_task.add_done_callback(_log_detached_play)
                        play_task.add_done_callback(release_playback_lock)
                        total_elapsed = time.monotonic() - started
                        logger.info(
                            "⏱️ HomePod router: retour prudent après %.2fs sans état "
                            "RAOP exploitable (play_media maintenu en arrière-plan; "
                            "%d car.)",
                            total_elapsed,
                            len(text),
                        )
                        return True, total_elapsed

                await play_task
            finally:
                if not detached_play:
                    if not probe_task.done():
                        probe_task.cancel()
                    try:
                        await probe_task
                    except asyncio.CancelledError:
                        pass

            play_elapsed = time.monotonic() - play_started_at
            total_elapsed = time.monotonic() - started

            transition = None
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
                "⏱️ HomePod router: verrou=%.2fs état=%.2fs volume=%.2fs%s url=%.2fs "
                "play_media=%.2fs demarrage_raop=%s total=%.2fs (%d car.) vers %s",
                lock_wait,
                state_elapsed,
                volume_elapsed,
                " (sauté)" if volume_skipped else "",
                url_elapsed,
                play_elapsed,
                "%.2fs" % transition if transition is not None else "non détecté",
                total_elapsed,
                len(text),
                self._target_entity,
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
    finally:
        if not detached_play:
            release_playback_lock()


HomePodSpeechRouter.__init__ = _mc10_router_init
HomePodSpeechRouter._hold_seconds_for_text = _mc10_hold_seconds_for_text
HomePodSpeechRouter._speak_on_homepod = _mc10_speak_on_homepod

logger.info(
    "🚀 Maison Cognitive mc10 chargé: fillers 1.5s, réponses complètes 0.25s, "
    "continuation 2.0s, retour RAOP=%s, verrou AirPlay actif",
    os.environ.get("HOMEPOD_EARLY_RETURN_ON_RAOP", "true"),
)
