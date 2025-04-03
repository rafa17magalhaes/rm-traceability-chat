# RM Traceability Chat Microservice

Este repositório contém o **RM Traceability Chat Microservice**, um chatbot especializado em auxiliar os usuários a navegar e utilizar o sistema RM Traceability SaaS. O serviço utiliza o GPT4All para gerar respostas contextuais com base em informações do sistema, exemplos de perguntas/respostas e uma integração simples para recuperação de informações (RAG) e gerenciamento de sessões.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Overview

O RM Traceability Chat Microservice é um chatbot que responde perguntas específicas sobre o RM Traceability SaaS, um sistema para o gerenciamento de empresas com foco na rastreabilidade de produtos via QR Code. Ele foi desenvolvido utilizando FastAPI e GPT4All e conta com recursos como:

- **Gerenciamento de Sessão:** Histórico de conversa armazenado por `session_id`.
- **Few-Shot Learning:** Exemplos de perguntas e respostas para guiar o modelo.
- **Integração RAG Simples:** Recupera informações relevantes dos serviços do sistema.
- **REST API:** Para fácil integração com outros sistemas (por exemplo, com seu backend NestJS).

## Features

- **Contextual Chatbot:** Gera respostas baseadas no contexto do sistema e nos exemplos fornecidos.
- **Gerenciamento de Sessão:** Mantém o histórico de conversa para diálogos multi-turno.
- **Exemplos de Few-Shot:** Inclui vários exemplos de perguntas e respostas para orientar o modelo.
- **RAG Simples:** Incorpora informações relevantes sobre os serviços do sistema ao prompt, quando aplicável.
- **API RESTful:** Desenvolvido com FastAPI para alta performance e facilidade de integração.

## Architecture

O projeto foi estruturado de forma modular para facilitar a manutenção e escalabilidade:

- **src/main.py:** Ponto de entrada do FastAPI, onde as rotas são registradas.
- **src/services/chat_service.py:** Contém a lógica de geração de respostas, gerenciamento de histórico e integração com o GPT4All.
- **src/models/chat_request.py:** Define os modelos Pydantic para validação dos dados de requisição.
- **system_context.txt:** Arquivo com informações contextuais detalhadas sobre o RM Traceability SaaS.
- **Dockerfile & docker-compose.yml:** Arquivos para containerização e deploy do micro serviço.

## Installation

### Prerequisites

- [Python 3.9+](https://www.python.org/downloads/) (para execução local)
- [Docker](https://www.docker.com/) (para containerização)
- [Git](https://git-scm.com/)

### Local Setup

1. **Clone o repositório:**
   ```bash```
   git clone https://github.com/rafa17magalhaes/rm-traceability-chat.git
   cd rm-traceability-chat

   Crie e ative um ambiente virtual

   **Crie e ative um ambiente virtual:**

    python -m venv .venv
    .venv\Scripts\activate   # No Windows
    # ou
    source .venv/bin/activate  # Linux/Mac

  **Instale as dependências:**
   pip install -r requirements.txt

  **Configure as variáveis de ambiente: Crie um arquivo chamado .env baseado em .env.example**

### Docker Setup
  **Construa e execute o container usando Docker Compose**

  docker-compose up --build

## Testing
**Os testes unitários estão na pasta tests/. Para executá-los, com o ambiente virtual ativado, use:**
 pytest

## Project Structure

    ``` bash
    chat-microservice/
    ├── .env.example          # Exemplo de arquivo de variáveis de ambiente
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── system_context.txt    # Contexto do sistema para o chatbot
    ├── src
    │   ├── main.py           # Ponto de entrada do FastAPI
    │   ├── models
    │   │   └── chat_request.py  # Modelo Pydantic para a requisição
    │   └── services
    │       └── chat_service.py  # Lógica do chatbot e integração com GPT4All
    └── tests
        └── test_chat.py      # Testes unitários para o endpoint /chat
    ```

## License
**Este projeto é licenciado sob a MIT License.**

## Contact
- **Autor:** Rafael Magalhaes
Para dúvidas ou suporte, entre em contato com Rafael Magalhaes via rafa7magalhaes@outlook.com ou abra uma issue neste repositório.
