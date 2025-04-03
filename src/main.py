import os
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv

from src.services.chat_service import ChatService
from src.models.chat_request import ChatRequest

# Carrega variáveis do arquivo .env
load_dotenv()

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    session_id: str = Query(None, description="ID da sessão para histórico da conversa")
):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")
    
    response_text, new_session_id = ChatService.generate_response(user_message, session_id)
    return {"response": response_text, "session_id": new_session_id}
