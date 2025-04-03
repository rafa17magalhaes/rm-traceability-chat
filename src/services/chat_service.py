import os
import uuid
from fastapi import HTTPException
from gpt4all import GPT4All

# Dicionário para armazenar histórico de conversas por session_id
SESSIONS = {}

# Dicionário com informações dos serviços para integração RAG simples
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
    try:
        with open("system_context.txt", "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Contexto do sistema não disponível.")

# Carrega o nome do arquivo do modelo das variáveis de ambiente (default se não definido)
MODEL_FILE = os.getenv("MODEL_FILE", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
_model = GPT4All(MODEL_FILE)

def get_service_info(user_message: str) -> str:
    info = []
    for key, text in SERVICE_INFO.items():
        if key.lower() in user_message.lower():
            info.append(text)
    return "\n".join(info) if info else ""

class ChatService:
    @classmethod
    def generate_response(cls, user_message: str, session_id: str = None) -> (str, str):  # type: ignore
        # Se session_id não for fornecido, gera um novo
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Recupera histórico da sessão, se existir
        history = SESSIONS.get(session_id, [])
        
        # Exemplos para few-shot com múltiplos casos:
        examples = (
            "Usuário: Como acessar o inventário?\n"
            "Assistente: Para acessar o inventário no RM Traceability SaaS, navegue até a rota '/dashboard/codigos/inventory'. "
            "Nessa seção, você poderá visualizar e gerenciar o estoque de produtos, atualizar quantidades e conferir detalhes dos itens.\n\n"
            "Usuário: Como gerenciar empresas?\n"
            "Assistente: Para gerenciar empresas, acesse a rota '/dashboard/empresas', onde você pode cadastrar e editar informações das empresas e dos usuários.\n\n"
            "Usuário: Como registrar movimentação de produtos?\n"
            "Assistente: Para registrar movimentação de produtos, utilize a rota '/dashboard/codigos/movements', que permite registrar entradas e saídas de produtos."
        )
        
        # Recupera informações relevantes dos serviços (RAG simples)
        service_info = get_service_info(user_message)
        
        context = load_system_context()
        history_text = "\n".join(history)
        
        prompt = (
            f"### Contexto do Sistema:\n{context}\n\n"
            f"### Exemplos:\n{examples}\n\n"
            f"### Informações Relevantes:\n{service_info}\n\n"
            f"### Histórico da Conversa:\n{history_text}\n\n"
            f"### Instrução:\n"
            f"Você é um assistente especializado no RM Traceability SaaS. Responda exclusivamente com informações sobre este sistema, "
            f"utilizando o contexto, os exemplos e as informações relevantes fornecidos acima. Não mencione outros sistemas ou jogos.\n\n"
            f"Usuário: {user_message}\n"
            f"Assistente:"
        )
        
        try:
            with _model.chat_session() as chat:
                # max_tokens para 256 para respostas mais rápidas
                response_text = chat.generate(prompt, max_tokens=256)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {e}")
        
        # Atualiza o histórico da sessão (pergunta e resposta)
        history.append(f"Usuário: {user_message}")
        history.append(f"Assistente: {response_text}")
        SESSIONS[session_id] = history
        
        return response_text, session_id
