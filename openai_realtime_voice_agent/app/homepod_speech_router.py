"""Route every assistant spoken segment to the Salon HomePod.

MAISON COGNITIVE
=================

Target architecture:

- Voice PE Cuisine = microphone / wake word / LED / session control.
- HomePod Salon = conversational loudspeaker.
- OpenAI Realtime keeps generating audio normally.
- The spoken transcript is sent automatically to Home Assistant via
  ``script.mc_reponse_homepod``.
- Native OpenAI PCM is held as a fail-safe instead of being played immediately
  by the Voice PE.
- If Home Assistant accepts the HomePod TTS request, the held PCM is discarded.
- If HomePod routing fails, the held PCM is released to the Voice PE so the user
  is never left with silence.

The router also mirrors a successful HomePod playback into Pipecat's normal bot
speaking lifecycle with ``BotStartedSpeakingFrame`` / ``BotStoppedSpeakingFrame``.
The pipeline places this router BEFORE ``PhaseEmitter`` so the Voice PE still
receives ``replying`` / ``idle`` phases even though no native PCM reaches its
speaker.

Home Assistant's REST service call returns when TTS has been accepted, not when
the HomePod has physically finished speaking. We therefore keep the synthetic
``replying`` phase active for a conservative text-duration estimate. The timing
can be tuned with environment variables without changing code.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class HomePodSpeechRouter(FrameProcessor):
    """Route assistant speech to the Salon HomePod with Voice PE fallback."""

    def __init__(
        self,
        *,
        target_entity: str = "media_player.salon_salon_homepod",
        script_service: str = "mc_reponse_homepod",
        ha_api_base: str = "http://supervisor/core/api",
        timeout_seconds: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._target_entity = target_entity
        self._script_service = script_service
        self._ha_api_base = ha_api_base.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)

        self._response_active = False
        self._text_parts: list[str] = []
        self._audio_frames: list[OutputAudioRawFrame] = []

        # A very small PCM tail can arrive after LLMFullResponseEndFrame on some
        # versions/stacks. Once HomePod routing succeeds, keep dropping that tail
        # until the next LLMFullResponseStartFrame resets the response state.
        self._drop_audio_tail = False

        # Home Assistant returns before physical TTS playback completes. Keep the
        # Voice PE in `replying` for an estimated audible duration so its mic and
        # follow-up window do not reopen while the HomePod is still talking.
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

    def _reset_response(self) -> None:
        self._text_parts = []
        self._audio_frames = []
        self._drop_audio_tail = False

    def _get_ha_token(self) -> str:
        """Return the HA token available inside the add-on."""
        return (
            os.environ.get("LONGLIVED_TOKEN", "").strip()
            or os.environ.get("SUPERVISOR_TOKEN", "").strip()
        )

    def _call_home_assistant_sync(self, text: str) -> None:
        """Call script.mc_reponse_homepod through the Supervisor HA proxy."""
        token = self._get_ha_token()
        if not token:
            raise RuntimeError(
                "aucun jeton Home Assistant disponible "
                "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
            )

        url = f"{self._ha_api_base}/services/script/{self._script_service}"
        body = json.dumps(
            {
                "target": self._target_entity,
                "message": text,
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

    async def _speak_on_homepod(self, text: str) -> bool:
        """Ask Home Assistant to start TTS on the HomePod."""
        try:
            await asyncio.to_thread(self._call_home_assistant_sync, text)
            logger.info(
                "🏠 HomePod router: segment accepté par Home Assistant "
                f"vers {self._target_entity} ({len(text)} caractères)"
            )
            return True
        except Exception as exc:
            logger.warning(
                "⚠️ HomePod router: échec du routage, "
                f"fallback Voice PE activé: {exc!r}"
            )
            return False

    def _estimate_playback_seconds(self, text: str) -> float:
        """Estimate HomePod TTS duration conservatively.

        Google Translate / ordinary conversational French is roughly around
        14-16 characters per second including spaces. Add startup latency plus
        small punctuation pauses, then clamp the result so a malformed/huge
        response cannot lock the Voice PE in replying indefinitely.
        """
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

    async def _mirror_homepod_playback(
        self,
        text: str,
        direction: FrameDirection,
    ) -> None:
        """Drive the existing PhaseEmitter with normal bot-speaking frames.

        The pipeline deliberately places this router before PhaseEmitter. These
        synthetic system frames therefore trigger the exact same replying/idle
        logic as native Voice PE playback, without touching PhaseEmitter itself.
        """
        duration = self._estimate_playback_seconds(text)
        logger.info(
            "🗣️ HomePod router: phase replying estimée "
            f"pendant {duration:.1f}s"
        )

        await self.push_frame(BotStartedSpeakingFrame(), direction)
        try:
            await asyncio.sleep(duration)
        finally:
            await self.push_frame(BotStoppedSpeakingFrame(), direction)

    async def _release_voice_pe_fallback(
        self,
        direction: FrameDirection,
    ) -> None:
        """Release held native PCM to the Voice PE after routing failure."""
        buffered_frames = self._audio_frames
        self._audio_frames = []

        logger.warning(
            "🔊 HomePod router: diffusion de secours sur Voice PE "
            f"({len(buffered_frames)} trames PCM)"
        )

        for audio_frame in buffered_frames:
            await self.push_frame(audio_frame, direction)

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        # Only alter the assistant's downstream output. Upstream transport
        # speaking frames (used by native fallback) must pass unchanged.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._response_active = True
            self._reset_response()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TTSTextFrame, LLMTextFrame)):
            if self._response_active and frame.text:
                self._text_parts.append(frame.text)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OutputAudioRawFrame):
            if self._response_active:
                # Hold native Realtime PCM as a fail-safe until HomePod routing
                # success/failure is known at the response end.
                self._audio_frames.append(frame)
                return

            if self._drop_audio_tail:
                return

            # Unknown/unbracketed audio: preserve the original behavior rather
            # than risking silent loss.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._text_parts).strip()
            self._response_active = False

            if not text:
                logger.warning(
                    "⚠️ HomePod router: réponse sans texte exploitable, "
                    "fallback Voice PE"
                )
                await self._release_voice_pe_fallback(direction)
                await self.push_frame(frame, direction)
                self._reset_response()
                return

            routed = await self._speak_on_homepod(text)

            if routed:
                # Home Assistant accepted TTS. Suppress native PE audio and mirror
                # the audible HomePod interval into the standard speaking frames
                # so PhaseEmitter keeps LED/mic/follow-up state coherent.
                self._audio_frames = []
                self._drop_audio_tail = True
                logger.info(
                    "🔇 HomePod router: audio Voice PE supprimé "
                    "pour cette réponse"
                )
                await self._mirror_homepod_playback(text, direction)
            else:
                # HA/HomePod routing failed before acceptance: release native PCM.
                # The real output transport will then generate its own
                # BotStarted/Stopped frames, exactly as before this router existed.
                await self._release_voice_pe_fallback(direction)
                self._drop_audio_tail = False

            await self.push_frame(frame, direction)
            self._text_parts = []
            return

        await self.push_frame(frame, direction)
