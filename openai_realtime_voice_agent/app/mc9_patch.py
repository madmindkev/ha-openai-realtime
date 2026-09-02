"""Maison Cognitive mc9 runtime refinements.

This module is imported explicitly from root/run.sh before app.main starts.
It keeps the validated mc8 HomePod/Morgan routing intact and changes only two
behaviours:

1. Lower conversational start latency without bringing back chopped replies:
   - syntactically incomplete fragments wait 2.0 s instead of 3.2 s so a real
     continuation can still be merged (the live split we observed was ~1.7 s),
   - common pre-tool fillers such as "je vérifie" still keep a full 1.0 s guard,
   - ordinary complete short replies use a much shorter 0.35 s guard,
   - post-tool complete answers still bypass the guard entirely.

2. Reopen the Voice PE follow-up window immediately when a successfully routed
   Morgan/HomePod playback really ends. The normal PhaseEmitter debounce remains
   untouched for every other BotStoppedSpeakingFrame (including Voice PE
   fallback), so this optimisation is targeted and does not weaken fallback
   safety.
"""

import logging
import time

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from app.homepod_speech_router import HomePodSpeechRouter
from app.phase_emitter import PhaseEmitter, TURN_LIVENESS

logger = logging.getLogger("app.homepod_speech_router")
phase_logger = logging.getLogger("app.phase_emitter")

# Exact BotStopped frame identities generated after a successful HomePod playback.
# PhaseEmitter consumes the same frame object downstream and can therefore open
# follow-up immediately without affecting unrelated/fallback BotStopped frames.
_IMMEDIATE_IDLE_STOP_IDS: set[int] = set()

_ORIGINAL_ROUTER_INIT = HomePodSpeechRouter.__init__
_ORIGINAL_PHASE_PROCESS = PhaseEmitter.process_frame
_ORIGINAL_IDLE_AFTER_DEBOUNCE = PhaseEmitter._emit_idle_after_debounce

_TOOL_FILLER_PREFIXES = (
    "je vérifie",
    "je verifie",
    "je regarde",
    "je consulte",
    "je cherche",
    "je récupère",
    "je recupere",
    "je vais vérifier",
    "je vais verifier",
    "je vais regarder",
    "une seconde",
    "un instant",
    "attends",
    "laisse-moi",
    "laissez-moi",
    "vérifions",
    "verifions",
    "checking",
    "let me check",
    "let me look",
    "one moment",
)


def _mc9_router_init(self, *args, **kwargs):
    _ORIGINAL_ROUTER_INIT(self, *args, **kwargs)

    # Keep mc12's effective defaults while allowing the add-on options to reach
    # the attributes used by the final hold policy.
    self._pretool_hold_seconds = self._env_float(
        "HOMEPOD_PRETOOL_HOLD_SECONDS", 0.75, minimum=0.0
    )
    self._continuation_hold_seconds = self._env_float(
        "HOMEPOD_CONTINUATION_HOLD_SECONDS", 2.0, minimum=0.0
    )
    self._mc9_complete_hold_seconds = 0.35
    self._mc9_tool_filler_hold_seconds = self._env_float(
        "HOMEPOD_TOOL_FILLER_HOLD_SECONDS", 1.5, minimum=0.0
    )


def _looks_like_tool_filler(text: str) -> bool:
    clean = " ".join((text or "").strip().lower().split())
    return any(clean.startswith(prefix) for prefix in _TOOL_FILLER_PREFIXES)


def _mc9_hold_seconds_for_text(self, text: str, *, post_tool: bool) -> float:
    # A visibly unfinished fragment needs enough time for the next Realtime chunk
    # to arrive; the live problematic split was around 1.7 s, hence a 2.0 s guard.
    if self._looks_incomplete(text) and len(text) <= self._continuation_hold_max_chars:
        return self._continuation_hold_seconds

    # Once a tool has completed, a complete answer is useful final speech: no
    # anti-tool grace delay is needed.
    if post_tool:
        return 0.0

    if len(text) <= self._pretool_hold_max_chars:
        # Preserve the full race window for the fillers OpenAI sometimes emits
        # immediately before a tool call; ordinary complete speech can start much
        # sooner without sacrificing the mc4/mc6 filler suppression behaviour.
        if _looks_like_tool_filler(text):
            return self._mc9_tool_filler_hold_seconds
        return self._mc9_complete_hold_seconds

    return 0.0


async def _mc9_route_with_replying_phase(self, text, direction):
    """mc8 routing + an exact marker for successful HomePod playback end."""
    estimated = self._estimate_playback_seconds(text)
    await self.push_frame(BotStartedSpeakingFrame(), direction)
    routed = False
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
            import asyncio
            await asyncio.sleep(remaining)
        else:
            logger.info(
                "🗣️ HomePod router: appel HA a déjà couvert la lecture "
                f"({call_elapsed:.1f}s >= estimation {estimated:.1f}s), "
                "aucune attente supplémentaire"
            )
        if routed and self._follow_up_callback is not None:
            try:
                # Le Voice PE ne joue volontairement aucun PCM local quand
                # Morgan est diffusé sur le HomePod. Demander explicitement
                # son follow-up mic évite que le firmware interprète le tour
                # comme une réponse sans audio et exige un nouveau wake word.
                await self._follow_up_callback()
                logger.info(
                    "🔁 HomePod router: follow-up Voice PE demandé après "
                    "la réponse Morgan"
                )
            except Exception as exc:
                logger.warning(
                    "⚠️ HomePod router: impossible de demander le follow-up: %r",
                    exc,
                )
        return True
    finally:
        stop_frame = BotStoppedSpeakingFrame()
        if routed:
            _IMMEDIATE_IDLE_STOP_IDS.add(id(stop_frame))
        await self.push_frame(stop_frame, direction)


async def _mc9_phase_process(self, frame, direction):
    if isinstance(frame, BotStoppedSpeakingFrame) and id(frame) in _IMMEDIATE_IDLE_STOP_IDS:
        _IMMEDIATE_IDLE_STOP_IDS.discard(id(frame))
        # Tell the patched debounce coroutine that this particular stop is the end
        # of a successfully completed HomePod stream, not an intermediate native
        # TTS segment or a Voice PE fallback stop.
        self._mc9_immediate_idle_once = True
    return await _ORIGINAL_PHASE_PROCESS(self, frame, direction)


async def _mc9_idle_after_debounce(self):
    if getattr(self, "_mc9_immediate_idle_once", False):
        self._mc9_immediate_idle_once = False
        if TURN_LIVENESS.in_flight > 0:
            # Defensive parity with the original debounce behaviour. Normally a
            # final Morgan reply reaches here with no tool left in flight.
            await self._emit("thinking")
            self._arm_watchdog()
            return
        phase_logger.info(
            "🎤 HomePod terminé -> idle immédiat, fenêtre follow-up ouverte sans debounce"
        )
        await self._emit("idle")
        return

    await _ORIGINAL_IDLE_AFTER_DEBOUNCE(self)


HomePodSpeechRouter.__init__ = _mc9_router_init
HomePodSpeechRouter._hold_seconds_for_text = _mc9_hold_seconds_for_text
HomePodSpeechRouter._route_with_replying_phase = _mc9_route_with_replying_phase
PhaseEmitter.process_frame = _mc9_phase_process
PhaseEmitter._emit_idle_after_debounce = _mc9_idle_after_debounce

logger.info(
    "🚀 Maison Cognitive mc9 chargé: continuation 2.0s, réponse complète 0.35s, "
    "fillers outil 1.0s, follow-up HomePod immédiat"
)
