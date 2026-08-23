"""Route assistant speech to the Salon HomePod.

MAISON COGNITIVE
=================

Target architecture:

- Voice PE Cuisine = microphone / wake word / LED / session control.
- HomePod Salon = conversational loudspeaker.
- OpenAI Realtime keeps generating audio normally.
- Conversational speech is rendered by the dedicated Maison Cognitive Morgan
  TTS entity and played on the Salon HomePod through Home Assistant ``tts.speak``.
- Native OpenAI PCM is held as a fail-safe instead of being played immediately
  by the Voice PE.
- If Home Assistant accepts the HomePod TTS request, the held PCM is discarded.
- If HomePod routing fails, the held PCM is released to the Voice PE so the user
  is never left with silence.

Maison Cognitive refinements:

1. Short assistant replies are held for a brief grace window before HomePod
   playback. If a tool call starts during that window, the pending text/audio is
   discarded as pre-tool filler ("je vérifie", "une seconde", etc.).
2. The response that follows a tool call is marked as post-tool and bypasses the
   grace window, so the useful final answer is spoken without an extra delay.
3. Empty/duplicate LLM response-end markers are ignored quietly instead of
   producing misleading fallback warnings.
4. ``replying`` begins BEFORE the Home Assistant TTS call. Some TTS/media-player
   stacks block until playback has largely/fully completed; the measured HTTP
   call duration is therefore subtracted from the estimated playback duration so
   we never wait for the same speech twice.

The grace window defaults to 1.0 s for replies up to 160 characters and can be
adjusted with HOMEPOD_PRETOOL_HOLD_SECONDS / HOMEPOD_PRETOOL_HOLD_MAX_CHARS.
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

    def __init__(
        self,
        *,
        target_entity: str = "media_player.salon_salon_homepod",
        tts_entity: str = "tts.maison_cognitive_morgan_maison_cognitive_morgan",
        ha_api_base: str = "http://supervisor/core/api",
        timeout_seconds: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

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

        # A FunctionCallsStartedFrame can arrive just AFTER the preceding LLM end
        # marker (SystemFrames may overtake queued data/control frames). Keep a
        # short reply pending long enough to catch that race before speaking it.
        self._pretool_hold_seconds = self._env_float(
            "HOMEPOD_PRETOOL_HOLD_SECONDS", 1.0, minimum=0.0
        )
        self._pretool_hold_max_chars = self._env_int(
            "HOMEPOD_PRETOOL_HOLD_MAX_CHARS", 160, minimum=1
        )
        self._pending_segment: Optional[dict] = None
        self._pending_task: Optional[asyncio.Task] = None

        # The next LLM response after a tool call is the useful post-tool answer.
        # It bypasses the grace window so we do not add 1 s to the final result.
        self._next_response_is_post_tool = False

        # Drop a tiny PCM tail that can arrive just after an LLM end marker once
        # the corresponding speech is being held/routed elsewhere.
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

    def _call_home_assistant_sync(self, text: str) -> None:
        """Render speech with Maison Cognitive Morgan and play it on the HomePod."""
        token = self._get_ha_token()
        if not token:
            raise RuntimeError(
                "aucun jeton Home Assistant disponible "
                "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
            )

        # Modern Home Assistant TTS: target the configured TTS entity (which
        # carries the Fish Audio/Morgan voice settings) and provide the media
        # player separately. This avoids the legacy conversational script choosing
        # a different TTS provider.
        url = f"{self._ha_api_base}/services/tts/speak"
        body = json.dumps(
            {
                "entity_id": self._tts_entity,
                "media_player_entity_id": self._target_entity,
                "message": text,
                "cache": False,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                status = response.getcode()
                if status < 200 or status >= 300:
                    raise RuntimeError(f"Home Assistant HTTP {status}")
                response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Home Assistant HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Home Assistant inaccessible: {exc.reason}"
            ) from exc

    async def _speak_on_homepod(self, text: str) -> tuple[bool, float]:
        started = time.monotonic()
        try:
            await asyncio.to_thread(self._call_home_assistant_sync, text)
            elapsed = time.monotonic() - started
            logger.info(
                "🏠 HomePod router: Morgan TTS accepté par Home Assistant "
                f"via {self._tts_entity} vers {self._target_entity} "
                f"({len(text)} caractères, appel {elapsed:.1f}s)"
            )
            return True, elapsed
        except Exception as exc:
            elapsed = time.monotonic() - started
            logger.warning(
                "⚠️ HomePod router: échec Morgan TTS, "
                f"fallback Voice PE activé après {elapsed:.1f}s: {exc!r}"
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
                "🔊 HomePod router: diffusion de secours sur Voice PE "
                f"({len(audio_frames)} trames PCM)"
            )
        else:
            logger.debug("HomePod router: fallback sans PCM à restituer")

        for audio_frame in audio_frames:
            await self.push_frame(audio_frame, direction)

    def _cancel_pending_as_pretool(self, reason: str) -> bool:
        """Cancel a held short reply because a tool/follow-up response appeared."""
        segment = self._pending_segment
        if segment is None:
            return False

        task = self._pending_task
        self._pending_segment = None
        self._pending_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

        logger.info(
            "⏭️ HomePod router: pré-réponse avant outil supprimée "
            f"({len(segment['text'])} caractères, {len(segment['audio'])} trames PCM; {reason})"
        )
        return True

    def _schedule_short_reply(
        self,
        text: str,
        audio_frames: list[OutputAudioRawFrame],
        direction: FrameDirection,
    ) -> None:
        """Hold a short response briefly so a racing tool call can cancel it."""
        # There should normally be at most one pending reply. If a previous one
        # somehow survived until a new one is scheduled, prefer the newest and
        # suppress the older as an intermediate segment.
        self._cancel_pending_as_pretool("nouveau segment court arrivé")

        segment = {
            "text": text,
            "audio": audio_frames,
            "direction": direction,
        }
        self._pending_segment = segment
        self._drop_audio_tail = True
        self._pending_task = asyncio.create_task(self._deliver_pending_after_hold(segment))
        logger.info(
            "⏸️ HomePod router: réponse courte retenue "
            f"{self._pretool_hold_seconds:.1f}s pour détecter un éventuel outil "
            f"({len(text)} caractères)"
        )

    async def _deliver_pending_after_hold(self, segment: dict) -> None:
        try:
            await asyncio.sleep(self._pretool_hold_seconds)
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
            "▶️ HomePod router: aucun outil détecté pendant la fenêtre, "
            "diffusion de la réponse retenue"
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
            # A new response starting inside the grace window is another strong
            # signal that the held reply was only an intermediate/pre-tool turn.
            pending_cancelled = self._cancel_pending_as_pretool(
                "nouvelle réponse LLM pendant la fenêtre"
            )

            self._response_active = True
            self._reset_current_response()
            self._response_is_post_tool = (
                self._next_response_is_post_tool or pending_cancelled
            )
            self._next_response_is_post_tool = False
            self._drop_audio_tail = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, FunctionCallsStartedFrame):
            # This is the race mc3 could miss: the LLM end frame may already have
            # been processed, leaving the short filler pending. Cancel it even if
            # _response_active is False.
            self._cancel_pending_as_pretool("FunctionCallsStartedFrame")
            if self._response_active:
                self._response_had_tool_call = True
            self._next_response_is_post_tool = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TTSTextFrame, LLMTextFrame)):
            if self._response_active and frame.text:
                self._text_parts.append(frame.text)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OutputAudioRawFrame):
            if self._response_active:
                self._audio_frames.append(frame)
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

            # Let the context aggregator close the response immediately. HomePod
            # routing/fallback can happen after this without blocking tool frames.
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

            # The final answer after a tool call should be spoken immediately.
            # Ordinary short replies get the grace window because a tool-start
            # SystemFrame may still be racing just behind this end marker.
            should_hold = (
                not is_post_tool
                and self._pretool_hold_seconds > 0
                and len(text) <= self._pretool_hold_max_chars
            )
            if should_hold:
                self._schedule_short_reply(text, audio_frames, direction)
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
