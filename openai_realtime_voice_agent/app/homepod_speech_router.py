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
6. ``replying`` begins BEFORE the Home Assistant TTS call. The measured HTTP call
   duration is subtracted from the estimated playback duration so we never wait
   for the same speech twice.

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

        # Pre-tool race guard for ordinary short complete replies.
        self._pretool_hold_seconds = self._env_float(
            "HOMEPOD_PRETOOL_HOLD_SECONDS", 1.0, minimum=0.0
        )
        self._pretool_hold_max_chars = self._env_int(
            "HOMEPOD_PRETOOL_HOLD_MAX_CHARS", 160, minimum=1
        )

        # Longer, targeted guard only for text that looks unfinished. This is what
        # prevents "Maison Cognitive est l'intelligence de" from being spoken
        # before "la maison qui..." arrives a couple of seconds later.
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

        # Ignore closing quotes/brackets when checking sentence punctuation.
        terminal_probe = clean.rstrip(cls._CLOSING_CHARS).rstrip()
        if terminal_probe.endswith(cls._TERMINAL_PUNCTUATION):
            return False

        words = [w.strip(" ,;:!?().[]{}\"'«»”).“’").lower() for w in clean.split()]
        words = [w for w in words if w]
        if not words:
            return False

        if words[-1] in cls._CONTINUATION_WORDS:
            return True

        # Very short conversational answers are often complete even when the API
        # omitted punctuation ("bonjour", "c'est fait", "21 heures 30").
        if len(words) <= 3:
            return False

        # Longer text with no terminal punctuation is safer to treat as unfinished.
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

    def _call_home_assistant_sync(self, text: str) -> None:
        """Render speech with Maison Cognitive Morgan and play it on the HomePod."""
        token = self._get_ha_token()
        if not token:
            raise RuntimeError(
                "aucun jeton Home Assistant disponible "
                "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
            )

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
        # If something is already pending here, merge it rather than silently
        # replacing it. This protects against sentence/chunk boundaries that do
        # not come with a clean LLMFullResponseStartFrame.
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
        # Once text is pending, keep at least a tiny race guard even if the newly
        # merged text became complete. For ordinary text this naturally resolves
        # to the 1 s pre-tool guard; post-tool complete text can flush immediately.
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
            # If no tool was announced, a pending segment followed by another LLM
            # response is a continuation, not filler. Carry it into the new
            # response so both pieces are spoken as one utterance.
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
                # Some Realtime/Pipecat sequences emit a continuation text chunk
                # without a fresh FullResponseStartFrame. Merge it directly into
                # the pending utterance and restart the appropriate timer.
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
                # Keep continuation PCM too, so Voice PE fallback remains complete
                # if Morgan/Home Assistant routing eventually fails.
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
