"""
DocuMind — FastAPI application entrypoint.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.config import get_settings
from app.db.models import create_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

settings = get_settings()

app = FastAPI(
    title="DocuMind — Document Q&A API",
    description=(
        "Retrieval-Augmented Generation (RAG) chatbot. "
        "Upload PDFs or text files and ask questions in plain English."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents_router)
app.include_router(chat_router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    create_tables()
    logging.getLogger(__name__).info(
        "DocuMind started [provider=%s, model=%s]",
        settings.llm_provider,
        settings.llm_model if settings.llm_provider == "openai" else settings.anthropic_model,
    )


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "env": settings.app_env,
    }

