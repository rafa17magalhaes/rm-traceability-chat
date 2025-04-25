FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .env             ./
COPY download_model.py ./

# Baixa e padroniza o .gguf
RUN mkdir -p model && python download_model.py

COPY . .

EXPOSE 8000

ENV MODEL_FILE=/app/model/model.gguf
ENV BACKEND_URL=http://rm_traceability_app:3001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
