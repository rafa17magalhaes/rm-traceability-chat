import logging
from fastapi import APIRouter, HTTPException, Query

from src.models.chat_request import ChatRequest
from src.services.chat_service import ChatService

router = APIRouter()
logger = logging.getLogger("chat-api")

@router.post("/")
async def chat_endpoint(
    request: ChatRequest,
    session_id: str = Query(None, description="ID da sessão para histórico da conversa")
):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    final_session_id = request.session_id or session_id

    response_text, new_session_id = ChatService.generate_response(
        user_message=user_message,
        session_id=final_session_id,
        user_id=request.user_id,
        company_id=request.company_id
    )

    logger.info(f"Resposta gerada: {response_text}")
    logger.info(f"Session ID: {new_session_id}")

    return {"response": response_text, "session_id": new_session_id}
