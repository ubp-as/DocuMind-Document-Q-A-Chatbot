"""
Query logging service — persists every Q&A interaction to PostgreSQL.
"""

import logging
from sqlalchemy.orm import Session

from app.db.models import QueryLog

logger = logging.getLogger(__name__)


def log_query(
    db: Session,
    *,
    question: str,
    answer: str,
    sources: list[dict],
    model: str,
    tokens_used: int,
    latency_ms: int,
    top_k: int,
) -> QueryLog:
    """Write a query + response record to the query_logs table."""
    record = QueryLog(
        question=question,
        answer=answer,
        sources=sources,
        model=model,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        top_k=top_k,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("Logged query id=%s (%d ms, %d tokens)", record.id, latency_ms, tokens_used)
    return record


def get_query_history(db: Session, limit: int = 50, offset: int = 0) -> list[QueryLog]:
    return (
        db.query(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
