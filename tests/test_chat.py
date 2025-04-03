from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_chat_endpoint():
    payload = {"message": "Como acessar o inventário?"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "inventário" in data["response"].lower()
    assert "session_id" in data
