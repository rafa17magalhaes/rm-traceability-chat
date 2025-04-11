import logging
from fastapi import FastAPI
from dotenv import load_dotenv

from src.api.chat import router as chat_router

load_dotenv()

logger = logging.getLogger("chat-microservice")
logger.setLevel(logging.INFO)

def create_app() -> FastAPI:
    app = FastAPI(title="RM Traceability Chat Microservice")
    # Se quiser, configure middlewares, CORS, etc.
    
    # Inclui o router do Chat
    app.include_router(chat_router, prefix="/chat", tags=["Chat"])
    
    return app

app = create_app()
