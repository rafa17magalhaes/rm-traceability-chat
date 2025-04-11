import re
import logging

logger = logging.getLogger("product_extractor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def extract_product(frase: str) -> str:
    """
    Extrai o nome do produto da frase, considerando variações na estrutura:
    - "quantos/quantas X tenho no inventario/estoque"
    - "quantos/quantas X no inventário"
    """
    padrao = (
        r"(?i)(?:quant[oa]s?)\s+(.*?)\s+"
        r"(?:eu\s+tenho\s+)?"
        r"(?:no|em)(?:\s+meu)?\s+"
        r"(?:inventar(?:io|[íi]ario)|estoque)"
    )
    match = re.search(padrao, frase)
    if match:
        produto_extraido = match.group(1).strip()
        produto_extraido = re.sub(r"(?i)\btenho\b", "", produto_extraido).strip()
        logger.debug(f"[extract_product] Produto extraído: {produto_extraido}")
        return produto_extraido

    logger.debug("[extract_product] Nenhum produto extraído.")
    return ""
