"""
/api/v1/documents — upload, list, and delete documents.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Document, get_db
from app.services.ingestion import delete_document, ingest_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: UUID
    original_name: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    is_active: bool

    model_config = {"from_attributes": True}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF or plain-text file. The document is chunked, embedded,
    and stored in ChromaDB. Metadata is persisted to PostgreSQL.
    """
    doc = await ingest_document(file, db)
    return doc


@router.get("/", response_model=list[DocumentOut])
def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List all active indexed documents."""
    return (
        db.query(Document)
        .filter(Document.is_active == True)
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.delete("/{document_id}", status_code=204)
def remove_document(
    document_id: UUID,
    db: Session = Depends(get_db),
):
    """Remove a document from the index and storage."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_document(doc, db)
