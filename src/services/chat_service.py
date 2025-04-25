from __future__ import annotations
import os, re, uuid, logging, unicodedata
from typing import Optional, Dict, List
from fastapi import HTTPException

from src.services.ml_service     import MLService
from src.utils.context_loader    import load_system_context
from src.utils.product_extractor import extract_product, _strip_accents
from src.utils.llm               import get_model
from src.config.constants        import SERVICE_INFO, EXAMPLES

logger = logging.getLogger("chat_service")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

SESSIONS: Dict[str, Dict] = {}
MODEL_FILE = os.getenv("MODEL_FILE", "/app/model/model.gguf")

def _chatml(system: str, user: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system\n"
        f"{system}\n<|end_header_id|>\n\n"
        "<|start_header_id|>user\n"
        f"{user}\n<|end_header_id|>\n\n"
        "<|start_header_id|>assistant\n"
    )

def _service_info_for(msg: str) -> str:
    return "\n".join(txt for k, txt in SERVICE_INFO.items() if k in msg.lower())

class ChatService:
    @classmethod
    def generate_response(
        cls,
        user_message: str,
        session_id: Optional[str] = None,
        user_id:     Optional[str] = None,
        company_id:  Optional[str] = None,
    ) -> tuple[str, str]:

        if not session_id:
            session_id = str(uuid.uuid4())
        sess  = SESSIONS.setdefault(session_id, {"history": [], "last_product": None})
        hist  = sess["history"]
        produto_ant = sess["last_product"]

        lower_no_acc = _strip_accents(user_message)

        if not company_id and user_id:
            company_id = MLService.get_company_id_for_user(user_id)

        produto = extract_product(user_message) or produto_ant

        if not produto and hist and lower_no_acc.startswith("e "):
            cand = lower_no_acc[2:].strip(" ?")
            if cand.isalpha():
                produto = cand
                sess["last_product"] = produto
                logger.debug("Heurística 'e <prod>': %s", produto)

        if produto:
            sess["last_product"] = produto

        if produto and company_id and (
            re.search(r"\b(quantos?|qtd|tem|tenho)\b", lower_no_acc) or
            lower_no_acc.startswith("e ")
        ):
            reply = MLService.fetch_inventory_for_product(produto, company_id).strip()
            hist += [f"Usuário: {user_message}", f"Assistente: {reply}"]
            return reply, session_id

        if re.search(r"\bcodigos?\b", lower_no_acc) and company_id:
            codes = MLService.fetch_codes_for_product(produto, company_id) if produto else []
            reply = (
                f"Códigos para '{produto}':\n" + "\n".join(codes)
                if produto and codes else
                "Nenhum código encontrado." if produto else
                "Desculpe, não consegui identificar o produto."
            )
            hist += [f"Usuário: {user_message}", f"Assistente: {reply}"]
            return reply, session_id

        system_ctx = (
            f"{load_system_context()}\n\n{EXAMPLES}\n\n"
            f"{_service_info_for(user_message)}\n\n"
            "⚠️ Responda em **português do Brasil** sem repetir a pergunta."
        )
        if produto and company_id:
            inv = MLService.fetch_inventory_for_product(produto, company_id)
            user_message += f"\n{inv}\n[Responda **apenas** com base nesses dados.]"

        prompt = _chatml(system_ctx, user_message)

        try:
            model = get_model()
            with model.chat_session() as chat:
                raw = chat.generate(
                    prompt=prompt,
                    max_tokens=64,
                    temp=0.15,
                    top_p=0.85,
                    repeat_penalty=1.15
                )
        except Exception as e:
            logger.exception("LLM failure")
            raise HTTPException(status_code=500, detail=f"Erro do modelo: {e}")

        reply = re.sub(r"<\|.*?\|>", "", raw or "").strip()
        if not reply:
            reply = (
                "Posso ajudar somente com o **RM Traceability SaaS** "
                "(inventário, códigos, empresas…), mas estou à disposição! 😉"
            )

        hist += [f"Usuário: {user_message}", f"Assistente: {reply}"]
        return reply, session_id
