import logging

logger = logging.getLogger("context_loader")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def load_system_context() -> str:
    """
    Lê o conteúdo de 'system_context.txt' para uso como contexto do sistema.
    """
    try:
        with open("system_context.txt", "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        logger.error("Erro ao carregar system_context.txt, usando contexto default.", exc_info=True)
        return "Contexto default."
