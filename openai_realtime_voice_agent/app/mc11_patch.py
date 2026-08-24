"""Maison Cognitive mc11 runtime refinements.

mc11 keeps the validated mc10 HomePod/Morgan/tool behaviour and targets the
remaining delay between the end of the user's speech and the first useful
assistant text.

Two changes only:

1. Semantic VAD latency trial: when the installed add-on is still configured at
   eagerness=low, mc11 promotes the effective runtime value to medium. "low" is
   deliberately the slowest semantic end-of-turn detector; medium should shorten
   the silence before OpenAI commits the user turn while remaining much safer
   than jumping straight to high. If the user already selected medium/high/auto,
   mc11 respects that value.

2. Timing instrumentation: log an approximate last-voiced-mic -> VAD-stop delay,
   then VAD-stop -> first assistant text/tool-call delay. Together with mc10's
   existing response-hold, Morgan URL and RAOP transition timings, one log now
   exposes the whole latency chain without changing the proven audio routing.

The mic "last voiced" estimate uses PCM RMS only for diagnostics. It never gates
or modifies audio sent to OpenAI.
"""

import audioop
import logging
import os
import time

import app.mc10_patch  # install the complete mc10 baseline first
from app.homepod_speech_router import HomePodSpeechRouter
from app.phase_emitter import PhaseEmitter
from app.websocket_handler import InputResampler
from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    LLMTextFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

logger = logging.getLogger("app.mc11")

# ---------------------------------------------------------------------------
# 1) Reduce semantic-VAD end-of-turn latency conservatively.
# ---------------------------------------------------------------------------
_configured_eagerness = os.environ.get("VAD_EAGERNESS", "low").strip().lower()
if _configured_eagerness == "low":
    os.environ["VAD_EAGERNESS"] = "medium"
    _effective_eagerness = "medium"
    logger.info(
        "⚡ mc11: semantic_vad eagerness runtime low -> medium "
        "(latency trial; HA saved option remains untouched)"
    )
else:
    _effective_eagerness = _configured_eagerness or "medium"
    logger.info(
        "⚡ mc11: semantic_vad eagerness conservé à %s",
        _effective_eagerness,
    )

try:
    _VOICE_RMS_THRESHOLD = max(
        1, int(os.environ.get("MC11_VOICE_RMS_THRESHOLD", "450"))
    )
except (TypeError, ValueError):
    _VOICE_RMS_THRESHOLD = 450

_TIMING = {
    "last_packet": 0.0,
    "last_voiced": 0.0,
    "vad_stop": 0.0,
    "first_assistant_logged": True,
    "first_tool_logged": True,
}

_ORIGINAL_RESAMPLER_PROCESS = InputResampler.process_frame
_ORIGINAL_PHASE_PROCESS = PhaseEmitter.process_frame
_ORIGINAL_ROUTER_PROCESS = HomePodSpeechRouter.process_frame


async def _mc11_resampler_process(self, frame, direction):
    """Observe mic timing/energy, then run the untouched input resampler."""
    if isinstance(frame, InputAudioRawFrame) and frame.audio:
        now = time.monotonic()
        _TIMING["last_packet"] = now
        try:
            rms = audioop.rms(frame.audio, 2)
        except Exception:
            rms = 0
        if rms >= _VOICE_RMS_THRESHOLD:
            _TIMING["last_voiced"] = now

    return await _ORIGINAL_RESAMPLER_PROCESS(self, frame, direction)


async def _mc11_phase_process(self, frame, direction):
    now = time.monotonic()

    if isinstance(frame, UserStartedSpeakingFrame):
        # New genuine turn. Start a fresh measurement even if the RMS estimator
        # has not crossed the voice threshold yet.
        _TIMING["last_voiced"] = now
        _TIMING["vad_stop"] = 0.0
        _TIMING["first_assistant_logged"] = False
        _TIMING["first_tool_logged"] = False

    elif isinstance(frame, UserStoppedSpeakingFrame):
        _TIMING["vad_stop"] = now
        _TIMING["first_assistant_logged"] = False
        _TIMING["first_tool_logged"] = False

        last_voiced = _TIMING.get("last_voiced", 0.0)
        last_packet = _TIMING.get("last_packet", 0.0)
        voice_gap = now - last_voiced if last_voiced else -1.0
        packet_gap = now - last_packet if last_packet else -1.0

        logger.info(
            "⏱️ mc11 VAD: dernier signal vocal -> fin de tour %.2fs "
            "(dernier paquet micro %.2fs, eagerness=%s, seuil RMS=%d)",
            voice_gap,
            packet_gap,
            _effective_eagerness,
            _VOICE_RMS_THRESHOLD,
        )

    return await _ORIGINAL_PHASE_PROCESS(self, frame, direction)


async def _mc11_router_process(self, frame, direction):
    now = time.monotonic()
    vad_stop = _TIMING.get("vad_stop", 0.0)

    if (
        vad_stop
        and not _TIMING.get("first_assistant_logged", False)
        and isinstance(frame, (TTSTextFrame, LLMTextFrame))
        and getattr(frame, "text", None)
    ):
        _TIMING["first_assistant_logged"] = True
        last_voiced = _TIMING.get("last_voiced", 0.0)
        logger.info(
            "⏱️ mc11 OpenAI: fin VAD -> premier texte assistant %.2fs "
            "(dernier signal vocal -> premier texte %.2fs)",
            now - vad_stop,
            now - last_voiced if last_voiced else -1.0,
        )

    if (
        vad_stop
        and not _TIMING.get("first_tool_logged", False)
        and isinstance(frame, FunctionCallsStartedFrame)
    ):
        _TIMING["first_tool_logged"] = True
        logger.info(
            "⏱️ mc11 OpenAI: fin VAD -> démarrage outil %.2fs",
            now - vad_stop,
        )

    return await _ORIGINAL_ROUTER_PROCESS(self, frame, direction)


InputResampler.process_frame = _mc11_resampler_process
PhaseEmitter.process_frame = _mc11_phase_process
HomePodSpeechRouter.process_frame = _mc11_router_process

logger.info(
    "🚀 Maison Cognitive mc11 chargé: mc10 intact, semantic_vad=%s, "
    "chronométrage voix->VAD->OpenAI actif",
    _effective_eagerness,
)
