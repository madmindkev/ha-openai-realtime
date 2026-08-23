"""Route every assistant spoken segment to the Salon HomePod.

MAISON COGNITIVE
=================
Architecture cible :

- Voice PE Cuisine = microphone / wake word / LED / contrôle de session.
- HomePod Salon = haut-parleur conversationnel.
- OpenAI Realtime continue de produire sa réponse normalement.
- Le texte réellement prononcé par OpenAI est envoyé automatiquement à
  Home Assistant -> script.mc_reponse_homepod.
- Le PCM OpenAI destiné au haut-parleur du Voice PE est retenu.
- Si le HomePod fonctionne, ce PCM est supprimé.
- Si le routage HomePod échoue, le PCM est relâché vers le Voice PE afin
  de ne jamais laisser l'utilisateur sans réponse.

Ce routeur fonctionne aussi pour les réponses intermédiaires avant un outil,
par exemple :

    "Je vérifie la température."

puis, après l'appel d'outil :

    "Il fait 24 degrés."

Chaque réponse OpenAI délimitée par LLMFullResponseStartFrame /
LLMFullResponseEndFrame est traitée indépendamment.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request

from pipecat.frames.frames import (
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
    """Route la parole de l'assistant vers le HomePod Salon.

    Le routeur collecte deux choses en parallèle :

    1. le texte prononcé par OpenAI :
       - TTSTextFrame en modalité audio Realtime ;
       - LLMTextFrame en secours pour une éventuelle modalité texte ;

    2. les OutputAudioRawFrame qui seraient normalement envoyées au
       haut-parleur du Voice PE.

    À la fin de chaque réponse OpenAI :

    - si le texte est disponible, le script Home Assistant
      `script.mc_reponse_homepod` est appelé ;
    - si cet appel réussit, le PCM Voice PE est abandonné ;
    - s'il échoue, le PCM retenu est relâché vers le Voice PE.
    """

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

        # Certaines piles peuvent laisser passer une toute petite queue PCM
        # après le EndFrame. Quand une réponse a été routée avec succès vers
        # le HomePod, cette queue appartient elle aussi à cette réponse et
        # ne doit pas ressortir sur le Voice PE.
        self._drop_audio_tail = False

    def _reset_response(self) -> None:
        """Réinitialise les tampons pour une nouvelle réponse."""
        self._text_parts = []
        self._audio_frames = []
        self._drop_audio_tail = False

    def _get_ha_token(self) -> str:
        """Récupère le jeton Home Assistant disponible dans l'add-on.

        L'add-on possède normalement SUPERVISOR_TOKEN grâce à :

            homeassistant_api: true

        LONGLIVED_TOKEN reste prioritaire lorsqu'il a été explicitement
        configuré.
        """
        return (
            os.environ.get("LONGLIVED_TOKEN", "").strip()
            or os.environ.get("SUPERVISOR_TOKEN", "").strip()
        )

    def _call_home_assistant_sync(self, text: str) -> None:
        """Appel REST synchrone exécuté dans un thread via asyncio.to_thread."""
        token = self._get_ha_token()

        if not token:
            raise RuntimeError(
                "aucun jeton Home Assistant disponible "
                "(LONGLIVED_TOKEN / SUPERVISOR_TOKEN)"
            )

        url = (
            f"{self._ha_api_base}/services/"
            f"script/{self._script_service}"
        )

        payload = {
            "target": self._target_entity,
            "message": text,
        }

        body = json.dumps(payload).encode("utf-8")

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
                    raise RuntimeError(
                        f"Home Assistant HTTP {status}"
                    )

                # Lire la réponse pour fermer proprement le flux HTTP.
                response.read()

        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Home Assistant HTTP {exc.code}"
            ) from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Home Assistant inaccessible: {exc.reason}"
            ) from exc

    async def _speak_on_homepod(self, text: str) -> bool:
        """Demande à Home Assistant de prononcer le texte sur le HomePod."""
        try:
            await asyncio.to_thread(
                self._call_home_assistant_sync,
                text,
            )

            logger.info(
                "🏠 HomePod router: segment envoyé avec succès "
                f"vers {self._target_entity} "
                f"({len(text)} caractères)"
            )

            return True

        except Exception as exc:
            logger.warning(
                "⚠️ HomePod router: échec du routage, "
                f"fallback Voice PE activé: {exc!r}"
            )

            return False

    async def _release_voice_pe_fallback(
        self,
        direction: FrameDirection,
    ) -> None:
        """Relâche le PCM retenu vers le Voice PE en cas d'échec HomePod."""
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

        # Le routeur ne modifie que la sortie descendante de l'assistant.
        # Tous les événements remontant vers OpenAI continuent normalement.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        # --------------------------------------------------------------
        # DÉBUT D'UNE RÉPONSE OPENAI
        # --------------------------------------------------------------
        if isinstance(frame, LLMFullResponseStartFrame):
            self._response_active = True
            self._reset_response()

            await self.push_frame(frame, direction)
            return

        # --------------------------------------------------------------
        # TEXTE PRONONCÉ
        # --------------------------------------------------------------
        if isinstance(frame, (TTSTextFrame, LLMTextFrame)):
            if self._response_active and frame.text:
                self._text_parts.append(frame.text)

            # Le texte continue dans le pipeline pour le logger,
            # le contexte et les autres processeurs.
            await self.push_frame(frame, direction)
            return

        # --------------------------------------------------------------
        # AUDIO OPENAI DESTINÉ AU VOICE PE
        # --------------------------------------------------------------
        if isinstance(frame, OutputAudioRawFrame):
            if self._response_active:
                # Ne pas envoyer immédiatement cet audio au Voice PE.
                # On le conserve comme solution de secours.
                self._audio_frames.append(frame)
                return

            if self._drop_audio_tail:
                # Petite queue audio appartenant à une réponse déjà
                # diffusée avec succès sur le HomePod.
                return

            # Audio hors réponse encadrée : comportement d'origine par sécurité.
            await self.push_frame(frame, direction)
            return

        # --------------------------------------------------------------
        # FIN D'UNE RÉPONSE OPENAI
        # --------------------------------------------------------------
        if isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._text_parts).strip()

            self._response_active = False

            if not text:
                # Sans transcription fiable, impossible d'envoyer du TTS
                # HomePod. On privilégie donc la sécurité : Voice PE.
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
                # Le HomePod a accepté la réponse :
                # le PCM OpenAI ne doit pas être joué sur le Voice PE.
                self._audio_frames = []
                self._drop_audio_tail = True

                logger.info(
                    "🔇 HomePod router: audio Voice PE supprimé "
                    "pour cette réponse"
                )

            else:
                # HomePod indisponible : la maison doit quand même répondre.
                await self._release_voice_pe_fallback(direction)
                self._drop_audio_tail = False

            await self.push_frame(frame, direction)

            self._text_parts = []
            return

        # --------------------------------------------------------------
        # TOUTES LES AUTRES TRAMES
        # --------------------------------------------------------------
        await self.push_frame(frame, direction)
