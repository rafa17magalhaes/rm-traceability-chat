from __future__ import annotations
import os
import re
import logging
import unidecode
import unicodedata
from typing import Optional
from functools import lru_cache
from threading import Lock
from gpt4all import GPT4All

# ──────────────────────────── logger ────────────────────────────
logger = logging.getLogger("product_extractor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)

# ───────────────────── funções auxiliares ───────────────────────
def _strip_accents(txt: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize("NFD", txt)
        if unicodedata.category(c) != "Mn"
    ).lower()

def _sanitize(txt: str) -> str:
    txt = unidecode.unidecode(txt.lower())
    txt = re.sub(r"[^a-z0-9\s\-]", "", txt).strip()
    if txt.endswith("s") and len(txt) > 3:
        txt = txt[:-1]
    return txt

# ───────────────────────── regex principal ──────────────────────
_REGEX = re.compile(
    r"(?i)"
    r"(?!.*\bnao sei\b)"
    r"(?:"
        r"quant(?:os|as)?|qtd(?:ade)?|tem|tenho|possui(?:mos)?|mostrar|ver|"
        r"(?:códigos?|codigos?)\s+(?:do|da|de|dos|das)"
    r")"
    r"(?:\s+(?:do|da|de|o|a|os|as))?\s+"
    r"(?P<prod>\w{2,})"
    r"(?:\s+(?:no|na|nos|nas|em|do|da|dos|das)\s+(?:meu|minha|nosso|nossa))?"
    r"(?:\s*(?:invent[aá]rio|estoque|produto))?"
)

# ────────────────────────── mini-modelo ─────────────────────────
_MODEL_FILE = os.getenv("MODEL_FILE", "/app/model/model.gguf")
_lock       = Lock()
_model: Optional[GPT4All] = None

def _get_model() -> GPT4All:
    global _model
    with _lock:
        if _model is None:
            logger.info("Carregando modelo tiny p/ extractor: %s", _MODEL_FILE)
            _model = GPT4All(_MODEL_FILE, n_threads=os.cpu_count() or 4)
        return _model

# ─────────────────────── função pública ─────────────────────────
@lru_cache(maxsize=1024)
def extract_product(sentence: str) -> str:
    if m := _REGEX.search(sentence):
        prod = _sanitize(m.group("prod"))
        logger.debug("Regex capturou: %s", prod)
        return prod

    prompt = (
        "Você receberá uma frase. "
        "Se pedir códigos ou quantidade de um item, responda **somente** o nome do item "
        "(singular, minúsculo, sem acento). Caso contrário responda \"NONE\".\n\n"
        f'Frase: "{sentence.strip()}"\nProduto:'
    )
    try:
        mdl = _get_model()
        with mdl.chat_session() as chat:
            raw = chat.generate(prompt=prompt, max_tokens=8, temp=0.2)
        logger.debug("Extractor LLM cru: %s", raw)
        if not raw or raw.upper().startswith("NONE"):
            return ""
        return _sanitize(raw)
    except Exception:
        logger.exception("Extractor LLM failure")
        return ""
