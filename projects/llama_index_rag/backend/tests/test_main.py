import pytest
from fastapi.testclient import TestClient

from backend.main import app
from rag_core.models import QueryResult, SourceInfo

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_success(monkeypatch):
    fake_result = QueryResult(
        answer="Mocked answer",
        sources=[
            SourceInfo(
                file_name="doc.pdf",
                score=0.91,
                snippet="relevant snippet",
                access_level="internal",
            )
        ],
        escalate=False,
    )

    def fake_ask_with_sources(question, user_role):
        return fake_result

    # патчим там, где функция ИСПОЛЬЗУЕТСЯ (в backend.main), а не там, где определена
    monkeypatch.setattr("backend.main.ask_with_sources", fake_ask_with_sources)

    response = client.post(
        "/query",
        json={"question": "test", "user_role": "internal"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer"
    assert body["sources"][0]["access_level"] == "internal"
    assert body["escalate"] is False


def test_query_invalid_role(monkeypatch):
    def fake_ask_with_sources(question, user_role):
        raise ValueError("Invalid user role: hacker")

    monkeypatch.setattr("backend.main.ask_with_sources", fake_ask_with_sources)

    response = client.post(
        "/query",
        json={"question": "test", "user_role": "hacker"},
    )

    assert response.status_code == 400
    assert "Invalid user role" in response.json()["detail"]


def test_query_missing_fields():
    # Pydantic-валидация тела запроса — до вызова ask_with_sources
    response = client.post("/query", json={"question": "test"})
    assert response.status_code == 422


def test_ingest(monkeypatch):
    def fake_ingest_documents():
        return {"status": "ok", "documents_ingested": 5}

    monkeypatch.setattr("backend.main.ingest_documents", fake_ingest_documents)

    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents_ingested": 5}