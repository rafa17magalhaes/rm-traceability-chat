from __future__ import annotations
import os, re, logging, unidecode
from functools import lru_cache
from threading import Lock
from gpt4all import GPT4All

logger = logging.getLogger("product_extractor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)

# ──────────────────────────────────────  REGEX fallback ──
_REGEX = re.compile(
    r"(?i)(?:quant(?:os|as)?|qtd(?:ade)?|tem|tenho|possui(?:mos)?|mostrar|ver)"
    r"\s+(?:de\s+|o\s+|a\s+|os\s+|as\s+)?"
    r"(?P<prod>.+?)"
    r"\s+(?:no|na|nos|nas|em)\s+(?:meu|minha|nosso|nossa)?\s*"
    r"(?:invent[aá]rio|estoque)"
)

def _sanitize(text: str) -> str:
    text = unidecode.unidecode(text.lower())
    text = re.sub(r"[^a-z0-9\s\-]", "", text).strip()
    if text.endswith("s") and len(text) > 3:
        text = text[:-1]
    return text

# ──────────────────────────────────────  LLM helper ──
_MODEL_FILE = os.getenv("MODEL_FILE", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
_model_lock = Lock()
_model: GPT4All | None = None   # será carregado no 1º uso

def _get_model() -> GPT4All:
    global _model
    with _model_lock:
        if _model is None:
            logger.info("Carregando modelo %s … isso leva alguns segundos", _MODEL_FILE)
            _model = GPT4All(_MODEL_FILE, n_threads=os.cpu_count() or 4)
        return _model

# ──────────────────────────────────────  Função pública ──
@lru_cache(maxsize=1024)
def extract_product(sentence: str) -> str:
    """Retorna produto em singular minúsculo ou ''."""
    if m := _REGEX.search(sentence):
        prod = _sanitize(m.group("prod"))
        logger.debug("Regex pegou: %s", prod)
        return prod

    # fallback LLM
    prompt = (
        "Você receberá uma frase.\n"
        "Se ela perguntar sobre quantidade em estoque ou inventário de ALGUM item, "
        "responda apenas com o nome desse item (singular, minúsculo, sem acento). "
        'Caso contrário, responda "NONE".\n\n'
        f'Frase: "{sentence.strip()}"\nProduto:'
    )

    try:
        model = _get_model()
        with model.chat_session() as chat:
            raw = chat.generate(prompt, max_tokens=10).strip()
        logger.debug("LLM cru: %s", raw)
        if raw.upper().startswith("NONE"):
            return ""
        return _sanitize(raw)
    except Exception:
        logger.exception("Erro na extração via LLM")
        return ""
