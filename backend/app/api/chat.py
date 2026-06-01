"""
/api/v1/chat — query and streaming endpoints.
"""

import json
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.rag_engine import run_rag_query, stream_rag_query
from app.db.models import get_db
from app.services.query_logger import get_query_history, log_query

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceOut(BaseModel):
    document: str
    page: int
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str
    tokens_used: int
    latency_ms: int


class QueryLogOut(BaseModel):
    id: UUID
    question: str
    answer: str
    sources: list[dict]
    model: str
    tokens_used: int
    latency_ms: int

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Run a RAG query and return a JSON response with answer + source citations.
    The interaction is logged to PostgreSQL.
    """
    result = await run_rag_query(body.question, top_k=body.top_k)

    log_query(
        db,
        question=body.question,
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
        tokens_used=result["tokens_used"],
        latency_ms=result["latency_ms"],
        top_k=body.top_k,
    )

    return result


@router.post("/stream")
async def stream(body: QueryRequest):
    """
    Run a RAG query with streaming response via Server-Sent Events.
    Tokens are streamed as they're generated; a final __SOURCES__ event
    carries metadata (sources + latency).
    """

    async def event_generator():
        async for token in stream_rag_query(body.question, top_k=body.top_k):
            if token.startswith("__SOURCES__"):
                data = token[len("__SOURCES__"):]
                yield f"event: sources\ndata: {data}\n\n"
            else:
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history", response_model=list[QueryLogOut])
def history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Retrieve paginated query history from PostgreSQL."""
    return get_query_history(db, limit=limit, offset=offset)
