"""
Test suite for DocuMind API.

Uses an in-memory SQLite DB and mocked LLM/ChromaDB calls so tests
run without any API keys or external services.
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, get_db
from app.main import app

# ── Test DB (SQLite in-memory) ────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Document upload ───────────────────────────────────────────────────────────

SAMPLE_TXT = b"The quick brown fox jumps over the lazy dog. " * 50


@patch("app.services.ingestion.get_vector_store")
@patch("app.services.ingestion.reset_vector_store_cache")
def test_upload_txt(mock_reset, mock_vs, client):
    """Uploading a .txt file returns a 201 with document metadata."""
    # Mock ChromaDB add_documents
    mock_vs.return_value.add_documents = MagicMock(return_value=None)

    file = io.BytesIO(SAMPLE_TXT)
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file, "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["original_name"] == "test.txt"
    assert data["file_type"] == "txt"
    assert data["chunk_count"] > 0
    assert data["is_active"] is True


def test_upload_unsupported_type(client):
    """Uploading an unsupported file type returns 415."""
    file = io.BytesIO(b"data")
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("data.csv", file, "text/csv")},
    )
    assert resp.status_code == 415


# ── Document list ─────────────────────────────────────────────────────────────

@patch("app.services.ingestion.get_vector_store")
@patch("app.services.ingestion.reset_vector_store_cache")
def test_list_documents(mock_reset, mock_vs, client):
    mock_vs.return_value.add_documents = MagicMock(return_value=None)

    # Upload two docs
    for name in ("a.txt", "b.txt"):
        client.post(
            "/api/v1/documents/upload",
            files={"file": (name, io.BytesIO(SAMPLE_TXT), "text/plain")},
        )

    resp = client.get("/api/v1/documents/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── Chat query ────────────────────────────────────────────────────────────────

RAG_RESULT = {
    "answer": "The answer is 42.",
    "sources": [{"document": "test.txt", "page": 0, "excerpt": "...relevant text..."}],
    "model": "gpt-4o",
    "tokens_used": 200,
    "latency_ms": 500,
}


@patch("app.api.chat.run_rag_query", new_callable=AsyncMock, return_value=RAG_RESULT)
def test_chat_query(mock_rag, client):
    resp = client.post(
        "/api/v1/chat/query",
        json={"question": "What is the answer?", "top_k": 4},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The answer is 42."
    assert len(data["sources"]) == 1
    assert data["model"] == "gpt-4o"


@patch("app.api.chat.run_rag_query", new_callable=AsyncMock, return_value=RAG_RESULT)
def test_query_logged_to_db(mock_rag, client):
    """Each query should be persisted to the query_logs table."""
    client.post("/api/v1/chat/query", json={"question": "Hello?", "top_k": 4})
    resp = client.get("/api/v1/chat/history")
    assert resp.status_code == 200
    logs = resp.json()
    assert len(logs) == 1
    assert logs[0]["question"] == "Hello?"


def test_query_empty_question(client):
    resp = client.post("/api/v1/chat/query", json={"question": ""})
    assert resp.status_code == 422  # Pydantic validation error


# ── Query history ─────────────────────────────────────────────────────────────

def test_history_empty(client):
    resp = client.get("/api/v1/chat/history")
    assert resp.status_code == 200
    assert resp.json() == []
