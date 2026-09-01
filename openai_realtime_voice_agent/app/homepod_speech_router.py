"""Route assistant speech to the Salon HomePod.

MAISON COGNITIVE
=================

Target architecture:

- Voice PE Cuisine = microphone / wake word / LED / session control.
- HomePod Salon = conversational loudspeaker.
- OpenAI Realtime keeps generating audio normally.
- Conversational speech is rendered by the dedicated Maison Cognitive Morgan
  TTS entity and played on the Salon HomePod.
- Native OpenAI PCM is held as a fail-safe instead of being played immediately
  by the Voice PE.
- If Home Assistant accepts the HomePod TTS request, the held PCM is discarded.
- If HomePod routing fails, the held PCM is released to the Voice PE so the user
  is never left with silence.

Maison Cognitive refinements:

1. Short assistant replies are held for a brief grace window before HomePod
   playback. If a tool call starts during that window, the pending text/audio is
   discarded as pre-tool filler ("je vérifie", "une seconde", etc.).
2. Fragmented assistant output is merged before speech. A fragment that looks
   syntactically unfinished gets a longer continuation window; if more LLM text
   arrives during that window it is appended to the same pending utterance rather
   than spoken as a separate HomePod clip.
3. If a new LLM response starts while a non-tool segment is still pending, the
   pending text/audio is carried into that response and the two are merged.
4. The response that follows a tool call bypasses the ordinary 1 s pre-tool
   grace when it is already a complete sentence; incomplete post-tool fragments
   still get the continuation window so they cannot be spoken mid-sentence.
5. Empty/duplicate LLM response-end markers are ignored quietly.
6. Morgan playback uses a fast Home Assistant path first: /api/tts_get_url
   creates a lazy TTS stream URL, then media_player.play_media starts that stream
   on the Salon HomePod. This avoids waiting for the entire tts.speak action.
7. If the fast path fails, the previously validated tts.speak transport is used
   automatically; Voice PE PCM remains the final fallback.
8. ``replying`` begins BEFORE HomePod routing. The measured routing call duration
   is subtracted from the estimated playback duration so we never wait twice.

Tunables:
- HOMEPOD_PRETOOL_HOLD_SECONDS (default 1.0 s)
- HOMEPOD_PRETOOL_HOLD_MAX_CHARS (default 160)
- HOMEPOD_CONTINUATION_HOLD_SECONDS (default 3.2 s)
- HOMEPOD_CONTINUATION_HOLD_MAX_CHARS (default 420)
"""

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class HomePodSpeechRouter(FrameProcessor):
    """Route final assistant speech to the Salon HomePod with Voice PE fallback."""

    _TERMINAL_PUNCTUATION = (".", "!", "?", "…")
    _CLOSING_CHARS = '"\'»”’)]}'
    _CONTINUATION_WORDS = {
        "à", "au", "aux", "avec", "car", "comme", "dans", "de", "des", "du",
        "en", "et", "lorsque", "mais", "ou", "par", "parce", "pour", "que",
        "quand", "qui", "sans", "si", "sous", "sur", "dont",
        "a", "an", "and", "as", "at", "because", "but", "by", "for", "from",
        "if", "in", "of", "on", "or", "that", "the", "to", "when", "which",
        "with", "without",
    }

    def __init__(
        self,
        *,
        route_policy: str = "all",
        target_entity: str = "media_player.salon_salon_homepod",
        tts_entity: str = "tts.maison_cognitive_morgan_maison_cognitive_morgan",
        ha_api_base: str = "http://supervisor/core/api",
        timeout_seconds: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._route_policy = (route_policy or "all").strip().lower()
        if self._route_policy not in {"all", "post_tool_only"}:
            logger.warning(
                "⚠️ HomePod router: politique inconnue %r, utilisation de all",
                route_policy,
            )
            self._route_policy = "all"
        self._target_entity = target_entity
        self._tts_entity = tts_entity
        self._ha_api_base = ha_api_base.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)

        # Current LLM response state.
        self._response_active = False
        self._response_had_tool_call = False
        self._response_is_post_tool = False
        self._text_parts: list[str] = []
        self._audio_frames: list[OutputAudioRawFrame] = []

        # Pre-tool race guard for ordinary short complete replies.
        self._pretool_hold_seconds = self._env_float(
            "HOMEPOD_PRETOOL_HOLD_SECONDS", 1.0, minimum=0.0
        )
        self._pretool_hold_max_chars = self._env_int(
            "HOMEPOD_PRETOOL_HOLD_MAX_CHARS", 160, minimum=1
        )

        # Longer, targeted guard only for text that looks unfinished.
        self._continuation_hold_seconds = self._env_float(
            "HOMEPOD_CONTINUATION_HOLD_SECONDS", 3.2, minimum=0.0
        )
        self._continuation_hold_max_chars = self._env_int(
            "HOMEPOD_CONTINUATION_HOLD_MAX_CHARS", 420, minimum=1
        )

        self._pending_segment: Optional[dict] = None
        self._pending_task: Optional[asyncio.Task] = None

        # The next LLM response after a tool call is the useful post-tool answer.
        self._next_response_is_post_tool = False

        # Drop tiny native PCM tails while text is being held/routed elsewhere.
        self._drop_audio_tail = False

        self._tts_chars_per_second = self._env_float(
            "HOMEPOD_TTS_CHARS_PER_SECOND", 15.0, minimum=5.0
        )
        self._tts_startup_seconds = self._env_float(
            "HOMEPOD_TTS_STARTUP_SECONDS", 0.8, minimum=0.0
        )
        self._tts_min_seconds = self._env_float(
            "HOMEPOD_TTS_MIN_SECONDS", 1.4, minimum=0.2
        )
        self._tts_max_seconds = self._env_float(
            "HOMEPOD_TTS_MAX_SECONDS", 30.0, minimum=1.0
        )

    @staticmethod
    def _env_float(name: str, default: float, *, minimum: float) -> float:
        try:
            value = float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    @staticmethod
    def _env_int(name: str, default: int, *, minimum: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    @classmethod
    def _looks_incomplete(cls, text: str) -> bool:
        """Return True when text looks like a fragment that should be merged."""
        clean = (text or "").strip()
        if not clean:
            return False

        terminal_probe = clean.rstrip(cls._CLOSING_CHARS).rstrip()
        if terminal_probe.endswith(cls._TERMINAL_PUNCTUATION):
            return False

        words = [w.strip(" ,;:!?().[]{}\"'«»”).“’").lower() for w in clean.split()]
        words = [w for w in words if w]
        if not words:
            return False

        if words[-1] in cls._CONTINUATION_WORDS:
            return True

        if len(words) <= 3:
            return False

        return True

    @staticmethod
    def _join_text(left: str, right: str) -> str:
        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        if right[0] in ",.;:!?)]}»”’":
            return left + right
        return left + " " + right

    def _reset_current_response(self) -> None:
        self._text_parts = []
        self._audio_frames = []
        self._response_had_tool_call = False
        self._response_is_post_tool = False

    def _get_ha_token(self) -> str:
        return (
            os.environ.get("LONGLIVED_TOKEN", "").strip()
            or os.environ.get("SUPERVISOR_TOKEN", "").strip()
        )

    def _request_json_sync(self, url: str, body: dict, token: str) -> dict:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                    status = response.getcode()
                    payload = response.read()
                    if status < 200 or status >= 300:
                        raise RuntimeError(f"Home Assistant HTTP {status}")
                    break
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(f"Home Assistant HTTP {exc.code}")
                if exc.code != 502 or attempt == 2:
                    raise last_error from exc
                time.sleep(0.35 * (attempt + 1))
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Home Assistant inaccessible: {exc.reason}")
                if attempt == 2:
                    raise last_error from exc
                time.sleep(0.35 * (attempt + 1))
        else:
            raise last_error or RuntimeError("Home Assistant request failed")

        if not payload:
            return {}
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("réponse JSON Home Assistant invalide") from exc
        if not isinstance(data, dict):
            raise RuntimeError("réponse JSON Home Assistant inattendue")
        return data

    def _request_service_sync(
        self,
        domain: str,
        service: str,
        body: dict,
        token: str,
    ) -> None:
        request = urllib.request.Request(
            url=f"{self._ha_api_base}/services/{domain}/{service}",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                    status = response.getcode()
                    if status < 200 or status >= 300:
                        raise RuntimeError(f"Home Assistant HTTP {status}")
                    response.read()
                    return
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(f"Home Assistant HTTP {exc.code}")
                if exc.code != 502 or attempt == 2:
                    raise last_error from exc
                time.sleep(0.35 * (attempt + 1))
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Home Assistant inaccessible: {exc.reason}")
                if attempt == 2:
                    raise last_error from exc
                time.sleep(0.35 * (attempt + 1))
        raise last_error or RuntimeError("Home Assistant request failed")

    def _call_home_assistant_fast_sync(self, text: str) -> tuple[float, float]:
        """Create a lazy Morgan stream URL, then start it on the Salon HomePod."""
        token = self._get_ha_token()
        if not token:
            raise RuntimeError(
                "aucun jeton Home Assistant disponible "
                "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
            )

        started_url = time.monotonic()
        tts_data = self._request_json_sync(
            f"{self._ha_api_base}/tts_get_url",
            {
                "engine_id": self._tts_entity,
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
        self._request_service_sync(
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
        play_elapsed = time.monotonic() - started_play
        return url_elapsed, play_elapsed

    def _call_home_assistant_legacy_sync(self, text: str) -> None:
        """Validated fallback: render Morgan through Home Assistant tts.speak."""
        token = self._get_ha_token()
        if not token:
            raise RuntimeError(
                "aucun jeton Home Assistant disponible "
                "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
            )

        self._request_service_sync(
            "tts",
            "speak",
            {
                "entity_id": self._tts_entity,
                "media_player_entity_id": self._target_entity,
                "message": text,
                "cache": False,
            },
            token,
        )

    async def _speak_on_homepod(self, text: str) -> tuple[bool, float]:
        started = time.monotonic()

        try:
            url_elapsed, play_elapsed = await asyncio.to_thread(
                self._call_home_assistant_fast_sync,
                text,
            )
            elapsed = time.monotonic() - started
            logger.info(
                "🐟 HomePod router: URL Morgan prête en %.2fs via %s",
                url_elapsed,
                self._tts_entity,
            )
            logger.info(
                "🍎 HomePod router: play_media accepté en %.2fs vers %s",
                play_elapsed,
                self._target_entity,
            )
            logger.info(
                "⚡ HomePod router: fast path Morgan accepté "
                "(%d caractères, total %.2fs)",
                len(text),
                elapsed,
            )
            return True, elapsed
        except Exception as fast_exc:
            logger.warning(
                "↩️ HomePod router: fast path Morgan indisponible, "
                "fallback tts.speak: %r",
                fast_exc,
            )

        try:
            legacy_started = time.monotonic()
            await asyncio.to_thread(self._call_home_assistant_legacy_sync, text)
            legacy_elapsed = time.monotonic() - legacy_started
            elapsed = time.monotonic() - started
            logger.info(
                "🏠 HomePod router: Morgan TTS accepté par fallback tts.speak "
                "via %s vers %s (%d caractères, fallback %.1fs, total %.1fs)",
                self._tts_entity,
                self._target_entity,
                len(text),
                legacy_elapsed,
                elapsed,
            )
            return True, elapsed
        except Exception as exc:
            elapsed = time.monotonic() - started
            logger.warning(
                "⚠️ HomePod router: échec Morgan TTS, "
                "fallback Voice PE activé après %.1fs: %r",
                elapsed,
                exc,
            )
            return False, elapsed

    def _estimate_playback_seconds(self, text: str) -> float:
        punctuation_pause = sum(text.count(ch) for ch in ".!?;:") * 0.08
        estimated = (
            self._tts_startup_seconds
            + (len(text) / self._tts_chars_per_second)
            + punctuation_pause
        )
        return min(
            self._tts_max_seconds,
            max(self._tts_min_seconds, estimated),
        )

    async def _route_with_replying_phase(
        self,
        text: str,
        direction: FrameDirection,
    ) -> bool:
        estimated = self._estimate_playback_seconds(text)
        await self.push_frame(BotStartedSpeakingFrame(), direction)
        try:
            routed, call_elapsed = await self._speak_on_homepod(text)
            if not routed:
                return False

            remaining = max(0.0, estimated - call_elapsed)
            if remaining > 0.05:
                logger.info(
                    "🗣️ HomePod router: phase replying, "
                    f"appel HA {call_elapsed:.1f}s + reliquat estimé {remaining:.1f}s"
                )
                await asyncio.sleep(remaining)
            else:
                logger.info(
                    "🗣️ HomePod router: appel HA a déjà couvert la lecture "
                    f"({call_elapsed:.1f}s >= estimation {estimated:.1f}s), "
                    "aucune attente supplémentaire"
                )
            return True
        finally:
            await self.push_frame(BotStoppedSpeakingFrame(), direction)

    async def _release_voice_pe_fallback_frames(
        self,
        audio_frames: list[OutputAudioRawFrame],
        direction: FrameDirection,
    ) -> None:
        if audio_frames:
            logger.warning(
                "🔇 HomePod router: audio local supprimé pour conserver "
                "Morgan exclusivement (routage HomePod indisponible; "
                f"{len(audio_frames)} trames PCM)"
            )
        else:
            logger.debug("HomePod router: aucun audio local à supprimer")

    def _stop_pending_task(self) -> None:
        task = self._pending_task
        self._pending_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _cancel_pending_as_pretool(self, reason: str) -> bool:
        """Discard a held segment because a tool call has started."""
        segment = self._pending_segment
        if segment is None:
            return False

        self._pending_segment = None
        self._stop_pending_task()
        logger.info(
            "⏭️ HomePod router: pré-réponse avant outil supprimée "
            f"({len(segment['text'])} caractères, {len(segment['audio'])} trames PCM; {reason})"
        )
        return True

    def _take_pending_for_merge(self, reason: str) -> Optional[dict]:
        """Remove and return a held segment without discarding it."""
        segment = self._pending_segment
        if segment is None:
            return None
        self._pending_segment = None
        self._stop_pending_task()
        logger.info(
            "🔗 HomePod router: segment retenu fusionné avec la suite "
            f"({len(segment['text'])} caractères; {reason})"
        )
        return segment

    def _hold_seconds_for_text(self, text: str, *, post_tool: bool) -> float:
        incomplete = self._looks_incomplete(text)
        if incomplete and len(text) <= self._continuation_hold_max_chars:
            return self._continuation_hold_seconds
        if post_tool:
            return 0.0
        if len(text) <= self._pretool_hold_max_chars:
            return self._pretool_hold_seconds
        return 0.0

    def _should_route_response(self, *, post_tool: bool) -> bool:
        """Return whether this completed response should use HomePod TTS."""
        return self._route_policy == "all" or (
            self._route_policy == "post_tool_only" and post_tool
        )

    def _schedule_pending_reply(
        self,
        text: str,
        audio_frames: list[OutputAudioRawFrame],
        direction: FrameDirection,
        *,
        post_tool: bool,
        hold_seconds: float,
        reason: str,
    ) -> None:
        """Hold a response so tool-race or continuation text can be detected."""
        previous = self._take_pending_for_merge("nouveau segment à mettre en attente")
        if previous is not None:
            text = self._join_text(previous["text"], text)
            audio_frames = list(previous["audio"]) + list(audio_frames)
            post_tool = bool(previous.get("post_tool")) or post_tool
            hold_seconds = self._hold_seconds_for_text(text, post_tool=post_tool)

        segment = {
            "text": text,
            "audio": list(audio_frames),
            "direction": direction,
            "post_tool": post_tool,
            "hold_seconds": hold_seconds,
        }
        self._pending_segment = segment
        self._drop_audio_tail = True
        self._pending_task = asyncio.create_task(
            self._deliver_pending_after_hold(segment, hold_seconds)
        )
        logger.info(
            "⏸️ HomePod router: réponse retenue "
            f"{hold_seconds:.1f}s ({reason}, {len(text)} caractères)"
        )

    def _refresh_pending_hold(self, *, reason: str) -> None:
        segment = self._pending_segment
        if segment is None:
            return
        hold_seconds = self._hold_seconds_for_text(
            segment["text"], post_tool=bool(segment.get("post_tool"))
        )
        segment["hold_seconds"] = hold_seconds
        self._stop_pending_task()
        if hold_seconds <= 0:
            self._pending_task = asyncio.create_task(
                self._deliver_pending_after_hold(segment, 0.0)
            )
        else:
            self._pending_task = asyncio.create_task(
                self._deliver_pending_after_hold(segment, hold_seconds)
            )
        logger.info(
            "🔗 HomePod router: suite ajoutée à la réponse retenue, "
            f"nouvelle fenêtre {hold_seconds:.1f}s ({reason}, {len(segment['text'])} caractères)"
        )

    async def _deliver_pending_after_hold(self, segment: dict, hold_seconds: float) -> None:
        try:
            if hold_seconds > 0:
                await asyncio.sleep(hold_seconds)
        except asyncio.CancelledError:
            return

        if self._pending_segment is not segment:
            return

        self._pending_segment = None
        self._pending_task = None
        text = segment["text"]
        audio_frames = segment["audio"]
        direction = segment["direction"]

        logger.info(
            "▶️ HomePod router: fenêtre terminée, diffusion de la réponse fusionnée "
            f"({len(text)} caractères)"
        )
        routed = await self._route_with_replying_phase(text, direction)
        if routed:
            self._drop_audio_tail = True
            logger.info("🔇 HomePod router: audio Voice PE supprimé pour cette réponse")
        else:
            self._drop_audio_tail = False
            await self._release_voice_pe_fallback_frames(audio_frames, direction)

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            carried = None
            if self._pending_segment is not None:
                if self._next_response_is_post_tool:
                    self._cancel_pending_as_pretool(
                        "nouvelle réponse post-outil après FunctionCallsStartedFrame"
                    )
                else:
                    carried = self._take_pending_for_merge(
                        "nouvelle réponse LLM sans outil"
                    )

            self._response_active = True
            self._reset_current_response()
            self._response_is_post_tool = self._next_response_is_post_tool
            self._next_response_is_post_tool = False
            self._drop_audio_tail = False

            if carried is not None:
                self._text_parts = [carried["text"]]
                self._audio_frames = list(carried["audio"])
                self._response_is_post_tool = (
                    self._response_is_post_tool or bool(carried.get("post_tool"))
                )

            await self.push_frame(frame, direction)
            return

        if isinstance(frame, FunctionCallsStartedFrame):
            self._cancel_pending_as_pretool("FunctionCallsStartedFrame")
            if self._response_active:
                self._response_had_tool_call = True
            self._next_response_is_post_tool = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TTSTextFrame, LLMTextFrame)):
            if self._response_active and frame.text:
                self._text_parts.append(frame.text)
            elif self._pending_segment is not None and frame.text:
                segment = self._pending_segment
                segment["text"] = self._join_text(segment["text"], frame.text)
                self._refresh_pending_hold(reason="nouveau texte LLM")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OutputAudioRawFrame):
            if self._response_active:
                self._audio_frames.append(frame)
                return

            if self._pending_segment is not None:
                self._pending_segment["audio"].append(frame)
                return

            if self._drop_audio_tail:
                return

            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if not self._response_active:
                await self.push_frame(frame, direction)
                return

            text = "".join(self._text_parts).strip()
            audio_frames = self._audio_frames
            had_tool_call = self._response_had_tool_call
            is_post_tool = self._response_is_post_tool

            self._response_active = False
            self._reset_current_response()

            await self.push_frame(frame, direction)

            if had_tool_call:
                self._drop_audio_tail = True
                logger.info(
                    "⏭️ HomePod router: pré-réponse avant outil supprimée "
                    f"({len(text)} caractères, {len(audio_frames)} trames PCM; "
                    "outil détecté dans la même réponse)"
                )
                return

            if not text:
                if audio_frames:
                    logger.warning(
                        "⚠️ HomePod router: audio sans texte exploitable, "
                        "fallback Voice PE"
                    )
                    self._drop_audio_tail = False
                    await self._release_voice_pe_fallback_frames(audio_frames, direction)
                else:
                    logger.debug(
                        "HomePod router: fin de réponse vide ignorée "
                        "(aucun texte, aucun PCM)"
                    )
                return

            if not self._should_route_response(post_tool=is_post_tool):
                self._drop_audio_tail = False
                await self._release_voice_pe_fallback_frames(audio_frames, direction)
                logger.info(
                    "🔈 HomePod router: réponse conservée sur Voice PE "
                    "(politique %s)",
                    self._route_policy,
                )
                return

            hold_seconds = self._hold_seconds_for_text(text, post_tool=is_post_tool)
            if hold_seconds > 0:
                reason = (
                    "fragment incomplet / attente de continuation"
                    if self._looks_incomplete(text)
                    else "détection d'un éventuel outil"
                )
                self._schedule_pending_reply(
                    text,
                    audio_frames,
                    direction,
                    post_tool=is_post_tool,
                    hold_seconds=hold_seconds,
                    reason=reason,
                )
                return

            routed = await self._route_with_replying_phase(text, direction)
            if routed:
                self._drop_audio_tail = True
                logger.info("🔇 HomePod router: audio Voice PE supprimé pour cette réponse")
            else:
                self._drop_audio_tail = False
                await self._release_voice_pe_fallback_frames(audio_frames, direction)
            return

        await self.push_frame(frame, direction)
