"""Maison Cognitive runtime patch: fast Morgan TTS -> Salon HomePod.

Loaded automatically by Python at startup because Docker copies root/ to / and
runs `python3 -m app.main` from WORKDIR /.

mc7 keeps the proven mc6 HomePodSpeechRouter (tool-filler suppression, fragmented
reply merging, Voice PE fallback) untouched and only replaces its HomePod speech
transport:

1. POST /api/tts_get_url with the Maison Cognitive Morgan TTS entity. Home
   Assistant returns a lazy /api/tts_proxy/... stream URL immediately; it does
   not have to finish rendering the full utterance first.
2. POST media_player.play_media with that URL to the Salon HomePod.
3. If either fast-path request fails synchronously, call the original mc6
   tts.speak transport. Voice PE fallback therefore remains the final safety net.
"""

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request

from app.homepod_speech_router import HomePodSpeechRouter

_LOGGER = logging.getLogger("app.homepod_speech_router")
_ORIGINAL_SPEAK_ON_HOMEPOD = HomePodSpeechRouter._speak_on_homepod


def _post_json(router, url: str, body: dict, token: str) -> dict:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=router._timeout_seconds,
        ) as response:
            status = response.getcode()
            payload = response.read()
            if status < 200 or status >= 300:
                raise RuntimeError(f"Home Assistant HTTP {status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Home Assistant HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Home Assistant inaccessible: {exc.reason}") from exc

    if not payload:
        return {}
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("réponse JSON Home Assistant invalide") from exc
    if not isinstance(data, dict):
        raise RuntimeError("réponse JSON Home Assistant inattendue")
    return data


def _post_service(router, domain: str, service: str, body: dict, token: str) -> None:
    request = urllib.request.Request(
        url=f"{router._ha_api_base}/services/{domain}/{service}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=router._timeout_seconds,
        ) as response:
            status = response.getcode()
            if status < 200 or status >= 300:
                raise RuntimeError(f"Home Assistant HTTP {status}")
            response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Home Assistant HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Home Assistant inaccessible: {exc.reason}") from exc


def _fast_morgan_sync(router, text: str) -> tuple[float, float, str]:
    token = router._get_ha_token()
    if not token:
        raise RuntimeError(
            "aucun jeton Home Assistant disponible "
            "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
        )

    # Home Assistant's current TTS API accepts `engine_id` as either a legacy
    # engine or a modern tts.* entity. It returns a no-auth /api/tts_proxy stream.
    started_url = time.monotonic()
    tts_data = _post_json(
        router,
        f"{router._ha_api_base}/tts_get_url",
        {
            "engine_id": router._tts_entity,
            "message": text,
            "cache": False,
        },
        token,
    )
    url_elapsed = time.monotonic() - started_url

    media_url = str(tts_data.get("url") or "").strip()
    if not media_url:
        raise RuntimeError("/api/tts_get_url n'a renvoyé aucune URL")

    started_play = time.monotonic()
    _post_service(
        router,
        "media_player",
        "play_media",
        {
            "entity_id": router._target_entity,
            "media_content_id": media_url,
            "media_content_type": "music",
            "announce": True,
        },
        token,
    )
    play_elapsed = time.monotonic() - started_play
    return url_elapsed, play_elapsed, media_url


async def _mc7_speak_on_homepod(self, text: str) -> tuple[bool, float]:
    """Use lazy TTS streaming first; preserve mc6 tts.speak as fallback."""
    started = time.monotonic()
    try:
        url_elapsed, play_elapsed, _media_url = await asyncio.to_thread(
            _fast_morgan_sync,
            self,
            text,
        )
        elapsed = time.monotonic() - started
        _LOGGER.info(
            "🐟 HomePod router: URL Morgan prête en %.2fs via %s",
            url_elapsed,
            self._tts_entity,
        )
        _LOGGER.info(
            "🍎 HomePod router: play_media accepté en %.2fs vers %s",
            play_elapsed,
            self._target_entity,
        )
        _LOGGER.info(
            "⚡ HomePod router: fast path Morgan accepté (%d caractères, total %.2fs)",
            len(text),
            elapsed,
        )
        return True, elapsed
    except Exception as fast_exc:
        _LOGGER.warning(
            "↩️ HomePod router: fast path Morgan indisponible, fallback tts.speak: %r",
            fast_exc,
        )

    # The original mc6 transport is deliberately retained as the first fallback.
    # It has already been validated live with this exact Morgan TTS entity/HomePod.
    routed, _legacy_elapsed = await _ORIGINAL_SPEAK_ON_HOMEPOD(self, text)
    return routed, time.monotonic() - started


HomePodSpeechRouter._speak_on_homepod = _mc7_speak_on_homepod
_LOGGER.info("⚡ Maison Cognitive mc7: fast Morgan TTS patch chargé")
