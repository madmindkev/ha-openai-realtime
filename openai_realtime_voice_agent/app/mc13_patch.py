"""Maison Cognitive mc13 — proactive Gmail bridge.

Adds a tiny loopback HTTP API to the existing OpenAI Realtime backend so Home
Assistant automations can hand a proactive Gmail context to the Voice PE stack.

Flow:
1. Home Assistant plays its proactive jingle + announcement on the Salon HomePod.
2. HA POSTs mail metadata to /mc/proactive/gmail.
3. This bridge injects the already-spoken question + private mail context into the
   current OpenAI Realtime conversation.
4. It sends {"type":"request_follow_up"} to the connected Voice PE.
5. The custom va_client firmware opens its microphone without a wake word.
6. The user's answer becomes the next normal Realtime turn, whose spoken response
   is already routed through Maison Cognitive Morgan -> Salon HomePod by mc8+.

The API binds to 127.0.0.1 by default, so it is not exposed to the LAN.
"""

import asyncio
import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pipecat.services.openai.realtime import events as openai_rt_events

import app.mc12_patch  # install the full mc12 -> ... -> mc9 stack first
from app.websocket_handler import WebSocketHandler

logger = logging.getLogger("app.mc13")

_ORIGINAL_WS_INIT = WebSocketHandler.__init__
_ORIGINAL_BUILD_PIPELINE = WebSocketHandler.build_pipeline

_BODY_LIMIT = 12000
_SUBJECT_LIMIT = 1000
_SENDER_LIMIT = 500
_QUESTION_LIMIT = 1000


def _clean(value: Any, limit: int) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return text[:limit]


def _mc13_ws_init(self, *args, **kwargs):
    _ORIGINAL_WS_INIT(self, *args, **kwargs)
    self._mc13_openai_service = None
    self._mc13_api_started = False
    self._mc13_api_task = None
    self._mc13_api_server = None
    self._mc13_proactive_item_ids = []


async def _delete_previous_proactive_items(handler: WebSocketHandler, service) -> None:
    previous = list(getattr(handler, "_mc13_proactive_item_ids", []) or [])
    handler._mc13_proactive_item_ids = []
    for item_id in previous:
        try:
            await service.send_client_event(
                openai_rt_events.ConversationItemDeleteEvent(item_id=item_id)
            )
        except Exception as exc:
            logger.debug(
                "Previous proactive item %s could not be deleted: %r", item_id, exc
            )


async def _inject_gmail_context(handler: WebSocketHandler, payload: dict) -> dict:
    service = getattr(handler, "_mc13_openai_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="OpenAI Realtime session not ready")

    connected = len(getattr(handler, "_websockets", []) or [])
    if connected < 1:
        raise HTTPException(status_code=503, detail="No Voice PE connected")

    sender = _clean(payload.get("sender"), _SENDER_LIMIT)
    subject = _clean(payload.get("subject"), _SUBJECT_LIMIT)
    body = _clean(payload.get("body"), _BODY_LIMIT)
    uid = _clean(payload.get("uid"), 200)
    message_id = _clean(payload.get("message_id"), 500)
    question = _clean(payload.get("question"), _QUESTION_LIMIT)

    if not sender:
        raise HTTPException(status_code=400, detail="sender is required")
    if not question:
        question = f"Nouveau mail de {sender}. Veux-tu la suite ?"

    await _delete_previous_proactive_items(handler, service)

    context_text = f"""
MAISON COGNITIVE — CONTEXTE GMAIL PROACTIF ACTIF

La Maison Cognitive vient d'annoncer oralement sur le HomePod :
"{question}"

L'utilisateur va répondre maintenant depuis le Voice PE, sans nouveau mot de réveil.

Mail actif :
- Expéditeur : {sender}
- Objet : {subject or "(sans objet)"}
- UID IMAP : {uid or "(inconnu)"}
- Message-ID : {message_id or "(inconnu)"}
- Contenu :
{body or "(contenu non fourni)"}

Règles pour CET échange :
- Interprète une réponse courte ("oui", "non", "vas-y", "plus tard", etc.) comme
  une réponse à la question Gmail ci-dessus.
- Si l'utilisateur accepte, donne d'abord uniquement l'objet du mail puis demande
  s'il veut un résumé, la lecture complète ou préparer une réponse.
- S'il demande un résumé, résume uniquement le contenu fourni ci-dessus.
- S'il demande la lecture complète, lis le contenu utile sans lire les en-têtes
  techniques ni les longues signatures répétitives.
- S'il demande de répondre, aide à rédiger oralement. Ne prétends jamais avoir
  envoyé le mail : aucun outil d'envoi Gmail n'est encore fourni par ce contexte.
- N'invente aucune information absente du mail.
- Ce contexte Gmail est le contexte proactif le plus récent et remplace tout
  ancien contexte Gmail proactif.
""".strip()

    context_item = openai_rt_events.ConversationItem(
        type="message",
        role="system",
        status="completed",
        content=[
            openai_rt_events.ItemContent(type="input_text", text=context_text)
        ],
    )
    question_item = openai_rt_events.ConversationItem(
        type="message",
        role="assistant",
        status="completed",
        content=[
            openai_rt_events.ItemContent(type="output_text", text=question)
        ],
    )

    await service.send_client_event(
        openai_rt_events.ConversationItemCreateEvent(item=context_item)
    )
    await service.send_client_event(
        openai_rt_events.ConversationItemCreateEvent(item=question_item)
    )
    handler._mc13_proactive_item_ids = [context_item.id, question_item.id]

    # The custom Voice PE firmware recognises this control frame and opens its
    # request-driven follow-up microphone lane without using HA assist_satellite.
    await handler.broadcast_json({"type": "request_follow_up"})

    logger.info(
        "📧 Proactive Gmail context armed for %s; Voice PE follow-up requested",
        sender,
    )
    return {
        "ok": True,
        "connected_devices": connected,
        "sender": sender,
        "subject_present": bool(subject),
        "body_chars": len(body),
    }


async def _run_proactive_api(handler: WebSocketHandler) -> None:
    host = os.environ.get("MC_PROACTIVE_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get("MC_PROACTIVE_API_PORT", "8091"))
    except (TypeError, ValueError):
        port = 8091

    api = FastAPI(
        title="Maison Cognitive Proactive Bridge",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @api.get("/health")
    async def health():
        service = getattr(handler, "_mc13_openai_service", None)
        return {
            "ok": True,
            "connected_devices": len(getattr(handler, "_websockets", []) or []),
            "realtime_session_ready": bool(
                service is not None and getattr(service, "_api_session_ready", False)
            ),
        }

    @api.post("/mc/proactive/gmail")
    async def proactive_gmail(payload: dict):
        return await _inject_gmail_context(handler, payload)

    config = uvicorn.Config(
        api,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    # This server runs as a background task inside the add-on's existing event
    # loop; the application lifecycle owns signals.
    server.install_signal_handlers = lambda: None
    handler._mc13_api_server = server

    logger.info("📡 Maison Cognitive proactive API listening on http://%s:%d", host, port)
    await server.serve()


def _mc13_build_pipeline(self, *args, **kwargs):
    # build_pipeline receives the live service as a named argument in current
    # main.py. Keep a reference for proactive context injection.
    service = kwargs.get("openai_service")
    if service is None and len(args) >= 2:
        service = args[1]
    self._mc13_openai_service = service

    result = _ORIGINAL_BUILD_PIPELINE(self, *args, **kwargs)

    if not getattr(self, "_mc13_api_started", False):
        self._mc13_api_started = True
        self._mc13_api_task = asyncio.create_task(_run_proactive_api(self))

    return result


WebSocketHandler.__init__ = _mc13_ws_init
WebSocketHandler.build_pipeline = _mc13_build_pipeline

logger.info(
    "🚀 Maison Cognitive mc13 chargé: bridge Gmail proactif + request_follow_up Voice PE"
)
