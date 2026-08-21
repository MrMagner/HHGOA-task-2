from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

def test_config_public(client: TestClient):
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert data["demo_mode"] is True

def test_text_query(client: TestClient):
    response = client.post("/api/v1/query", json={"query": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is RAG?"
    assert "answer" in data
    assert data["latency"]["total_ms"] > 0
