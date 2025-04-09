import os
import re
import uuid
from typing import Optional
from fastapi import HTTPException
import logging
from gpt4all import GPT4All

# Importa sua MLService para buscar inventory / companyId
from src.services.ml_service import MLService

logger = logging.getLogger("chat_service")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Dicionário de sessões de conversa, mantendo histórico em memória
SESSIONS = {}

# Dicionário com informações de serviços para "RAG" simples
SERVICE_INFO = {
    "inventario": "Inventário: Gerencie e visualize seu estoque. Acesse: /dashboard/codigos/inventory.",
    "empresas": "Empresas: Gerencie empresas e usuários. Acesse: /dashboard/empresas.",
    "movimentacoes": "Movimentar Produtos/Últimas Movimentações: Registre entradas e saídas e acompanhe alterações. Acesse: /dashboard/codigos/movements ou /dashboard/eventos.",
    "usuarios": "Usuários: Gerencie contas e permissões. Acesse: /dashboard/usuarios.",
    "produtos": "Produtos: Gerencie produtos da empresa. Acesse: /dashboard/recursos.",
    "codigos": "Códigos: Gerencie QR Codes. Acesse: /dashboard/codigos.",
    "geracaolote": "Geração em Lote: Gere múltiplos QR Codes de uma vez. Acesse: /dashboard/codigos/bulk-generate.",
    "status": "Status: Gerencie Status do sistema. Acesse: /dashboard/status.",
    "rastreamento": "Mapa de Rastreio: Visualize a localização das movimentações. Acesse: /dashboard/rastreamento.",
    "configuracoes": "Configurações: Gerencie dados do usuário e da empresa. Acesse: /dashboard/configuracoes."
}

def load_system_context() -> str:
    """
    Lê o conteúdo de 'system_context.txt' (se existir) para usar como contexto do sistema.
    """
    try:
        with open("system_context.txt", "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception:
        # Se não existir ou der erro, devolve um contexto default
        return "Contexto default."

def get_service_info(user_message: str) -> str:
    """
    Verifica se algumas palavras-chave aparecem na mensagem
    e retorna texto (RAG simples).
    """
    info = []
    for key, text in SERVICE_INFO.items():
        if key.lower() in user_message.lower():
            info.append(text)
    return "\n".join(info) if info else ""

def extract_product(frase: str) -> str:
    """
    Extrai o nome do produto da frase, considerando variações na estrutura:
    - quantos/quantas X tenho no inventario/estoque
    - quantos/quantas X no inventário
    - etc.
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
        # Se sobrar a palavra 'tenho' dentro do nome, removemos
        produto_extraido = re.sub(r"(?i)\btenho\b", "", produto_extraido).strip()
        logger.debug(f"[extract_product] Produto extraído: {produto_extraido}")
        return produto_extraido

    logger.debug("[extract_product] Nenhum produto extraído.")
    return ""

# Carrega o modelo GPT4All
MODEL_FILE = os.getenv("MODEL_FILE", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
_model = GPT4All(MODEL_FILE)

class ChatService:
    @classmethod
    def generate_response(
        cls,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        company_id: Optional[str] = None
    ) -> (str, str): # type: ignore
        """
        Gera a resposta do chat com base na mensagem do usuário e, se for caso de inventário,
        busca informações no BD via MLService.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Carrega histórico de conversa se existir
        history = SESSIONS.get(session_id, [])

        lower_msg = user_message.lower()

        if not company_id and user_id:
            logger.debug("[ChatService] Tentando descobrir company_id via user_id")
            found_company = MLService.get_company_id_for_user(user_id)
            if found_company:
                company_id = found_company
                logger.debug(f"[ChatService] company_id recuperado: {company_id}")
            else:
                logger.warning("[ChatService] Não foi possível obter company_id do usuário.")

        # Verifica se é caso de inventário:
        # Obs: você pode remover esse if e se basear só em extract_product(),
        # mas aqui mantemos para ilustrar.
        if "quantos" in lower_msg and ("inventário" in lower_msg or "inventario" in lower_msg):
            produto = extract_product(user_message)
            if produto and company_id:
                logger.debug(f"[ChatService] Produto={produto} e company_id={company_id}")
                inventory_answer = MLService.fetch_inventory_for_product(produto, company_id)
                # Injeta a resposta do BD no prompt
                user_message += (
                    "\n[IMPORTANTE: Estes dados do BD são verídicos e não devem ser alterados.]\n"
                    f"{inventory_answer}\n"
                    "[Responda SOMENTE com base nesses dados, sem contradizê-los.]\n"
                )
            else:
                logger.warning("[ChatService] Produto ou company_id ausente para inventário.")
        else:
            produto = extract_product(user_message)
            if produto and company_id:
                logger.debug(f"[ChatService] (via outro route) Produto={produto} e company_id={company_id}")
                inventory_answer = MLService.fetch_inventory_for_product(produto, company_id)
                user_message += (
                    "\n[IMPORTANTE: Estes dados do BD são verídicos e não devem ser alterados.]\n"
                    f"{inventory_answer}\n"
                    "[Responda SOMENTE com base nesses dados, sem contradizê-los.]\n"
                )

        # Se mensagem tiver "o que devo fazer agora" e user_id -> chama predição
        if "o que devo fazer agora" in lower_msg and user_id:
            ml_recommendation = MLService.predict_next_action(user_id)
            user_message += f"\n[Recomendação ML: {ml_recommendation}]"

        # Exemplos few-shot
        examples = (
            "Usuário: Como acessar o inventário?\n"
            "Assistente: Para acessar o inventário no RM Traceability SaaS, navegue até '/dashboard/codigos/inventory'.\n\n"
            "Usuário: Como gerenciar empresas?\n"
            "Assistente: Para gerenciar empresas, acesse '/dashboard/empresas'.\n\n"
            "Usuário: Como registrar movimentação de produtos?\n"
            "Assistente: Para registrar movimentação de produtos, utilize a rota '/dashboard/codigos/movements'."
        )

        # Carrega infos relevantes + contexto
        service_info = get_service_info(user_message)
        context = load_system_context()
        history_text = "\n".join(history)

        # Monta prompt
        prompt = (
            f"### Contexto do Sistema:\n{context}\n\n"
            f"### Exemplos:\n{examples}\n\n"
            f"### Informações Relevantes:\n{service_info}\n\n"
            f"### Histórico da Conversa:\n{history_text}\n\n"
            f"### Instrução:\n"
            f"Você é um assistente especializado no RM Traceability SaaS.\n\n"
            f"Usuário: {user_message}\n"
            f"Assistente:"
        )

        # Gera a resposta usando GPT4All
        try:
            with _model.chat_session() as chat:
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

        # Retorna a resposta + session_id
        return response_text, session_id
