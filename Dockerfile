FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de requisitos
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto para dentro do container
COPY . .

# Expõe a porta definida (8000 por padrão)
EXPOSE 8000

# Comando para rodar a aplicação com uvicorn (aponta para src/main.py)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
