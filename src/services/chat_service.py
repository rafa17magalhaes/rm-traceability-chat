import os
import re
import uuid
import logging
import unicodedata
from typing import Optional
from fastapi import HTTPException

from gpt4all import GPT4All
from src.services.ml_service import MLService
from src.utils.context_loader import load_system_context
from src.utils.product_extractor import extract_product
from src.config.constants import SERVICE_INFO, EXAMPLES

logger = logging.getLogger("chat_service")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)

# Em cada sessão guardamos histórico
SESSIONS: dict[str, dict] = {}

def _strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def get_service_info(user_message: str) -> str:
    info = []
    for key, text in SERVICE_INFO.items():
        if key.lower() in user_message.lower():
            info.append(text)
    return "\n".join(info) if info else ""

class ChatService:
    @classmethod
    def generate_response(
        cls,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str]    = None,
        company_id: Optional[str] = None,
    ) -> tuple[str, str]:
        # gera session_id se não veio
        if not session_id:
            session_id = str(uuid.uuid4())

        # carrega dados da sessão (history + last_product)
        session_data = SESSIONS.get(session_id, {"history": [], "last_product": None})
        history = session_data["history"]
        last_product = session_data["last_product"]

        # normaliza e strip accents
        lower_raw = user_message.lower()
        lower = _strip_accents(lower_raw)

        # tenta recuperar company_id
        if not company_id and user_id:
            cid = MLService.get_company_id_for_user(user_id)
            if cid:
                company_id = cid
                logger.debug(f"[ChatService] company_id recuperado: {company_id}")

        # tenta extrair produto explícito
        produto = extract_product(user_message)

        # se não extraiu e há referência indireta, reaproveita last_product
        if not produto and re.search(r"\b(ele|ela|este|esse|essa|isso|produto)\b", lower):
            produto = last_product
            if produto:
                logger.debug(f"[ChatService] Usando produto do contexto: {produto}")

        # se extraiu, atualiza o last_product
        if produto:
            session_data["last_product"] = produto

        # 1) Consulta EXPLÍCITA por códigos: detecta "codigos" sem acento
        if company_id and re.search(r"\bcodigos?\b", lower):
            if not produto:
                reply = "Desculpe, não consegui identificar o produto para listar códigos."
            else:
                codes = MLService.fetch_codes_for_product(produto, company_id)
                if codes:
                    reply = (
                        f"Esses são os códigos para o produto '{produto}' no inventário:\n"
                        + "\n".join(codes)
                    )
                else:
                    reply = f"Não encontrei códigos para o produto '{produto}' no inventário."
            history.extend([f"Usuário: {user_message}", f"Assistente: {reply}"])
            session_data["history"] = history
            SESSIONS[session_id] = session_data
            return reply, session_id

        # 2) Consulta quantidade em estoque
        if produto and company_id:
            logger.debug(f"[ChatService] Produto={produto} e company_id={company_id}")
            inv = MLService.fetch_inventory_for_product(produto, company_id)
            user_message += (
                "\n[IMPORTANTE: Estes dados do BD são verídicos e não devem ser alterados.]\n"
                f"{inv}\n[Responda SOMENTE com base nesses dados.]\n"
            )
        else:
            logger.warning("[ChatService] Produto ou company_id ausente para inventário.")

        # 3) Predição de próxima ação
        if "o que devo fazer agora" in lower and user_id:
            recomend = MLService.predict_next_action(user_id)
            user_message += f"\n[Recomendação ML: {recomend}]"

        # Monta prompt few‑shot e invoca LLM
        service_info = get_service_info(user_message)
        context      = load_system_context()
        history_text = "\n".join(history)
        prompt = (
            f"### Contexto do Sistema:\n{context}\n\n"
            f"### Exemplos:\n{EXAMPLES}\n\n"
            f"### Informações Relevantes:\n{service_info}\n\n"
            f"### Histórico da Conversa:\n{history_text}\n\n"
            "### Instrução:\nVocê é um assistente especializado no RM Traceability SaaS.\n\n"
            f"Usuário: {user_message}\nAssistente:"
        )

        MODEL_FILE = os.getenv("MODEL_FILE", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        try:
            with GPT4All(MODEL_FILE).chat_session() as chat:
                response = chat.generate(prompt, max_tokens=256)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {e}")

        history.extend([f"Usuário: {user_message}", f"Assistente: {response}"])
        session_data["history"] = history
        SESSIONS[session_id] = session_data

        return response, session_id
