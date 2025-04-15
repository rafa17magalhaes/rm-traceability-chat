import os
import uuid
from typing import Optional
from fastapi import HTTPException
import logging
from gpt4all import GPT4All

from src.services.ml_service import MLService
from src.utils.context_loader import load_system_context
from src.utils.product_extractor import extract_product
from src.config.constants import SERVICE_INFO, EXAMPLES

logger = logging.getLogger("chat_service")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Dicionário de sessões de conversa, mantendo histórico em memória
SESSIONS = {}

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
        user_id: Optional[str] = None,
        company_id: Optional[str] = None
    ) -> (str, str):  # type: ignore
        """
        Gera a resposta do chat com base na mensagem do usuário.
        Se extrair produto + tiver company_id, injeta dados de inventário do BD.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Carrega histórico de conversa se existir
        history = SESSIONS.get(session_id, [])

        lower_msg = user_message.lower()

        # Se company_id não foi passado, tenta descobrir a partir de user_id
        if not company_id and user_id:
            logger.debug("[ChatService] Tentando descobrir company_id via user_id")
            found_company = MLService.get_company_id_for_user(user_id)
            if found_company:
                company_id = found_company
                logger.debug(f"[ChatService] company_id recuperado: {company_id}")
            else:
                logger.warning("[ChatService] Não foi possível obter company_id do usuário.")

        # extrai o produto da mensagem
        produto = extract_product(user_message)
        if produto and company_id:
            logger.debug(f"[ChatService] Produto={produto} e company_id={company_id}")
            inventory_answer = MLService.fetch_inventory_for_product(produto, company_id)
            user_message += (
                "\n[IMPORTANTE: Estes dados do BD são verídicos e não devem ser alterados.]\n"
                f"{inventory_answer}\n"
                "[Responda SOMENTE com base nesses dados, sem contradizê-los.]\n"
            )
        else:
            logger.warning("[ChatService] Produto ou company_id ausente para inventário.")
        
        # Se mensagem tiver "o que devo fazer agora" e user_id -> chama predição
        if "o que devo fazer agora" in lower_msg and user_id:
            ml_recommendation = MLService.predict_next_action(user_id)
            user_message += f"\n[Recomendação ML: {ml_recommendation}]"

        # Carrega infos relevantes + contexto do sistema
        service_info = get_service_info(user_message)
        context = load_system_context()
        history_text = "\n".join(history)

        # Exemplos few-shot
        prompt = (
            f"### Contexto do Sistema:\n{context}\n\n"
            f"### Exemplos:\n{EXAMPLES}\n\n"
            f"### Informações Relevantes:\n{service_info}\n\n"
            f"### Histórico da Conversa:\n{history_text}\n\n"
            f"### Instrução:\n"
            f"Você é um assistente especializado no RM Traceability SaaS.\n\n"
            f"Usuário: {user_message}\n"
            f"Assistente:"
        )

        MODEL_FILE = os.getenv("MODEL_FILE", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        try:
            with GPT4All(MODEL_FILE).chat_session() as chat:
                response_text = chat.generate(prompt, max_tokens=256)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao gerar resposta: {e}"
            )

        # Atualiza histórico
        history.append(f"Usuário: {user_message}")
        history.append(f"Assistente: {response_text}")
        SESSIONS[session_id] = history

        return response_text, session_id
