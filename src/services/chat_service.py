import os
import re
import uuid
import logging
import unicodedata
from typing import Optional, Dict
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

class ChatService:
    FALLBACK = (
        "Desculpe, só posso ajudar com o **RM Traceability** "
        "(inventário, códigos, empresas…); 😉"
    )

    DOMAIN_INTENTS = [
        r"\b(quantos?|qtd|tem|tenho|possui|mostrar|ver)\b",
        r"\bcodig\w*\b",
        r"\binvent(?:ario|ario)\b",
        r"\bempresas?\b",
        r"\bmovimenta(?:coes|coes|cao)\b",
        r"\bprodutos?\b",
        r"\blotes?\b|\blote\b|\bgerar lote\b",
        r"\bstatus\b",
        r"\brastreia(?:mento|r)?\b|\bmapa\b",
        r"\bconfiguracoes?\b"
    ]

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
        sess = SESSIONS.setdefault(session_id, {"history": [], "last_product": None})
        hist = sess["history"]
        lower = _strip_accents(user_message)

        if not any(re.search(p, lower) for p in cls.DOMAIN_INTENTS):
            hist += [f"Usuário: {user_message}", f"Assistente: {cls.FALLBACK}"]
            return cls.FALLBACK, session_id

        if not company_id and user_id:
            company_id = MLService.get_company_id_for_user(user_id)

        if company_id and re.search(r"\b(quantos?|qtd|tem|tenho)\b", lower):
            produto = extract_product(user_message) or sess["last_product"]
            sess["last_product"] = produto or sess["last_product"]
            reply = MLService.fetch_inventory_for_product(produto, company_id).strip()
            hist += [f"Usuário: {user_message}", f"Assistente: {reply}"]
            return reply, session_id

        if company_id and re.search(r"\b(?:codig\w*|mostrar codig\w*|listar codig\w*|quais codig\w*)\b", lower):
            produto = extract_product(user_message) or sess["last_product"]
            sess["last_product"] = produto or sess["last_product"]
            codes = MLService.fetch_codes_for_product(produto, company_id)
            if produto and codes:
                reply = f"Códigos para '{produto}':\n" + "\n".join(codes)
            elif produto:
                reply = "Nenhum código encontrado."
            else:
                reply = "Desculpe, não consegui identificar o produto."
            hist += [f"Usuário: {user_message}", f"Assistente: {reply}"]
            return reply, session_id

        service_info_text = "\n".join(f"- {txt}" for txt in SERVICE_INFO.values())
        system_ctx = (
            f"{load_system_context()}\n\n"
            f"{EXAMPLES}\n\n"
            f"Informações de serviço:\n{service_info_text}\n\n"
            "⚠️ Responda em **português do Brasil** sem repetir a pergunta."
        )

        last_prod = sess["last_product"]
        if last_prod and company_id:
            inv = MLService.fetch_inventory_for_product(last_prod, company_id)
            user_message += f"\n{inv}\n[Responda **apenas** com base nesses dados.]"

        prompt = _chatml(system_ctx, user_message)

        try:
            model = get_model()
            with model.chat_session() as chat:
                raw = chat.generate(
                    prompt=prompt,
                    max_tokens=128,
                    temp=0.05,
                    top_p=0.5,
                    repeat_penalty=1.2
                )
        except Exception as e:
            logger.exception("LLM failure")
            raise HTTPException(status_code=500, detail=f"Erro do modelo: {e}")

        reply = re.sub(r"<\|.*?\|>", "", raw or "").strip()
        if not reply or re.match(r'^(none|nenhum|nao)\b', reply.lower()) or "/dashboard" not in reply:
            reply = cls.FALLBACK

        try:
            n_codes  = len(codes) if "codes" in locals() else 0
            n_events = sum(1 for m in hist if m.startswith("Usuário:"))
            next_act = MLService.predict_next_action(n_codes, n_events)
            logger.debug(f"Próxima ação sugerida: {next_act}")
        except Exception:
            logger.debug("Não foi possível predizer próxima ação", exc_info=True)

        hist += [f"Usuário: {user_message}", f"Assistente: {reply}"]
        return reply, session_id
