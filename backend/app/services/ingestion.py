"""
Ingestion service — loads, splits, and embeds uploaded documents into ChromaDB.
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.vector_store import get_vector_store, reset_vector_store_cache
from app.db.models import Document

logger = logging.getLogger(__name__)
settings = get_settings()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def ingest_document(file: UploadFile, db: Session) -> Document:
    """
    Full ingestion pipeline for an uploaded file:
      1. Validate file type + size
      2. Save to disk
      3. Load + split into chunks
      4. Embed + store in ChromaDB
      5. Persist metadata to PostgreSQL
    """
    _validate_file(file)

    # Save upload to disk
    ext = Path(file.filename).suffix.lower().lstrip(".")
    saved_name = f"{uuid.uuid4()}.{ext}"
    saved_path = UPLOAD_DIR / saved_name

    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB",
        )

    saved_path.write_bytes(content)
    logger.info("Saved upload: %s (%d bytes)", saved_path, len(content))

    # Load document
    docs = _load_document(str(saved_path), ext)
    if not docs:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Could not extract text from document")

    # Annotate metadata
    for doc in docs:
        doc.metadata["source"] = file.filename

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split '%s' into %d chunks", file.filename, len(chunks))

    # Assign unique IDs and embed into ChromaDB
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store = get_vector_store()
    vector_store.add_documents(chunks, ids=chunk_ids)
    reset_vector_store_cache()

    # Persist document record to PostgreSQL
    db_doc = Document(
        filename=saved_name,
        original_name=file.filename,
        file_type=ext,
        file_size_bytes=len(content),
        chunk_count=len(chunks),
        chroma_ids=chunk_ids,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    logger.info("Ingested document '%s' → %d chunks", file.filename, len(chunks))
    return db_doc


def delete_document(doc: Document, db: Session) -> None:
    """Remove a document from ChromaDB, disk, and PostgreSQL."""
    # Remove from ChromaDB
    if doc.chroma_ids:
        try:
            vector_store = get_vector_store()
            vector_store.delete(ids=doc.chroma_ids)
            reset_vector_store_cache()
        except Exception as e:
            logger.warning("Could not remove chunks from ChromaDB: %s", e)

    # Remove file from disk
    file_path = UPLOAD_DIR / doc.filename
    if file_path.exists():
        file_path.unlink()

    # Mark deleted in DB
    doc.is_active = False
    db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile) -> None:
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=415,
            detail=f"File type '.{ext}' not supported. Allowed: {settings.allowed_extensions}",
        )


def _load_document(path: str, ext: str):
    """Return a list of LangChain Documents from the file."""
    try:
        if ext == "pdf":
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        logger.error("Failed to load document '%s': %s", path, e)
        return []
