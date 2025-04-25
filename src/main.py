import logging
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from src.api.chat import router as chat_router
from src.utils.product_extractor import _get_model
from src.utils.llm               import get_model

load_dotenv()
logger = logging.getLogger("chat-microservice")
logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔄  Warm-up: carregando modelos…")
    _get_model()
    get_model()
    logger.info("✅  Modelos carregados.")
    yield

app = FastAPI(
    title="RM Traceability Chat Microservice",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])
