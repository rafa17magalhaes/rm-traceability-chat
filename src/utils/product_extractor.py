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
    padrao = (
        r"(?i)"                                   # Case-insensitive
        r"(?:tenho|tem|quant[oa]s?)\s+"           # 'tenho', 'tem', 'quanto', 'quantos', etc.
        r"(?:de\s+)?(.*?)\s+"                     # opcional "de " e depois produto (captura lazy)
        r"(?:no|em)(?:\s+meu)?\s+"                # 'no', 'em', 'no meu', 'em meu'
        r"(?:inventar(?:io|[íi]ario)|estoque)"    # 'inventario', 'estoque'
    )
    
    match = re.search(padrao, frase)
    if match:
        produto_extraido = match.group(1).strip()
        
        produto_extraido = re.sub(r"(?i)\b(tenho|tem)\b", "", produto_extraido).strip()
        
        produto_extraido = re.sub(r"(?i)^de\s+", "", produto_extraido).strip()

        if produto_extraido.endswith("s"):
            produto_extraido = produto_extraido[:-1]

        logger.debug(f"[extract_product] Produto extraído: {produto_extraido}")
        return produto_extraido

    logger.debug("[extract_product] Nenhum produto extraído.")
    return ""
