"""Maison Cognitive mc12 runtime refinements.

mc12 keeps the validated mc11 stack intact (medium semantic VAD, mc10 RAOP
instrumentation, mc9 immediate follow-up, Morgan/HomePod routing) and fixes the
remaining conversational rough edge: generic pre-tool speech that was not caught
by the known filler phrases.

Observed live with mc11:
    assistant: "D'accord, voyons la météo actuelle de la maison pour voir ça."
    ... ~0.43 s later ... FunctionCallsStartedFrame

Because that sentence is not literally "je vérifie" / "une seconde", mc10/mc11
released it after the normal 0.25 s guard and Morgan spoke it before the actual
tool result. That creates two HomePod replies for one user turn.

mc12 policy:
- incomplete fragments: keep the proven 2.0 s continuation window;
- known tool fillers: keep 1.5 s;
- ANY other complete short initial response: hold 0.75 s so a racing tool call
  can cancel it, regardless of wording;
- complete post-tool answers: 0 s (speak immediately);
- long complete replies (>220 chars): no extra generic hold.

0.75 s is intentionally targeted: the live generic-preamble -> tool-call gap was
~0.43 s, while a full 1.5 s on every ordinary answer would make normal dialogue
feel sluggish.

The patch also appends a small Maison Cognitive behavioural appendix to the
existing user prompt. It does not replace the saved prompt. The appendix tells the
Realtime model to treat the post-reply listening window as one continuous
conversation, to avoid spoken tool narration, to prefer one broad Home Assistant
live-context read over many redundant state reads when appropriate, and to use
agenda/weather/presence/vehicle context only when it actually helps the user.
Email is explicitly treated as a separate source: never invent mailbox knowledge
unless an email tool/context source is present.
"""

import logging
import os

import app.mc11_patch  # installs mc11 -> mc10 -> mc9 baseline first
from app.homepod_speech_router import HomePodSpeechRouter

logger = logging.getLogger("app.mc12")

# mc11 leaves the mc10 hold function installed. Replace only that policy.
_ORIGINAL_ROUTER_INIT = HomePodSpeechRouter.__init__


def _mc12_router_init(self, *args, **kwargs):
    _ORIGINAL_ROUTER_INIT(self, *args, **kwargs)
    self._mc12_generic_pretool_hold_seconds = 0.75
    self._mc12_generic_pretool_max_chars = 220
    # Preserve the values already validated in mc10/mc11.
    self._mc10_tool_filler_hold_seconds = 1.5
    self._continuation_hold_seconds = 2.0


def _mc12_hold_seconds_for_text(self, text: str, *, post_tool: bool) -> float:
    # A syntactically unfinished fragment still needs enough time for the next
    # Realtime chunk to arrive and be merged.
    if self._looks_incomplete(text) and len(text) <= self._continuation_hold_max_chars:
        return self._continuation_hold_seconds

    # Once a tool result is back, this is useful final speech. Do not add a
    # generic anti-tool delay to the answer the user is actually waiting for.
    if post_tool:
        return 0.0

    # mc10 exposes its normalized filler detector in the module. Import lazily
    # here to avoid duplicating/letting the phrase list drift.
    try:
        from app.mc10_patch import _looks_like_tool_filler
        if _looks_like_tool_filler(text):
            return self._mc10_tool_filler_hold_seconds
    except Exception:
        pass

    # Catch natural pre-tool phrases we cannot enumerate ("d'accord, voyons la
    # météo...", "regardons ça", etc.). A tool call arriving during this 750 ms
    # window cancels the held segment through the existing mc6+ race logic.
    if len(text) <= self._mc12_generic_pretool_max_chars:
        return self._mc12_generic_pretool_hold_seconds

    return 0.0


HomePodSpeechRouter.__init__ = _mc12_router_init
HomePodSpeechRouter._hold_seconds_for_text = _mc12_hold_seconds_for_text

# Add behaviour, never overwrite the user's carefully tuned prompt.
_COGNITIVE_APPENDIX = """

MAISON COGNITIVE — CONTINUITE ET CONTEXTE :
- Une fois reveille, considere les reponses suivantes de l'utilisateur comme la
  continuation naturelle de la meme conversation pendant la fenetre d'ecoute.
  Les reponses courtes comme « oui », « non », « et demain ? », « fais-le »,
  « pourquoi ? » se referent au contexte precedent sans exiger un nouveau wake word.
- Si un outil est necessaire, appelle-le silencieusement. Ne prononce pas de
  preambule du type « je verifie », « voyons ca », « une seconde » ou equivalent.
  Parle seulement lorsque tu as la reponse utile, sauf vraie question de clarification.
- Pour une question globale sur la maison, prefere GetLiveContext quand il suffit
  plutot que d'enchainer plusieurs lectures redondantes d'etat. Utilise ensuite un
  outil specialise seulement si une donnee plus precise ou plus fraiche est necessaire.
- Quand c'est pertinent, croise naturellement contexte maison, presence, agenda,
  meteo et vehicules pour donner un conseil concret. Ne recite pas un tableau de
  bord et ne signale pas des details sans utilite immediate.
- N'invente jamais le contenu des emails. Utilise les mails uniquement lorsqu'un
  outil ou une source email est effectivement disponible dans la session.
- Pour les actions reversibles et sans risque demandees clairement, agis directement
  puis confirme brievement. Pour une action ambigue ou sensible, demande confirmation.
""".strip()

_existing = os.environ.get("INSTRUCTIONS", "")
if "MAISON COGNITIVE — CONTINUITE ET CONTEXTE" not in _existing:
    os.environ["INSTRUCTIONS"] = (_existing.rstrip() + "\n\n" + _COGNITIVE_APPENDIX).strip()

logger.info(
    "🚀 Maison Cognitive mc12 chargé: pré-tool générique 0.75s, fillers 1.5s, "
    "continuation 2.0s, post-tool immédiat, mode conversation cognitive ajouté"
)
