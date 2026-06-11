"""
app/api/upload.py — Step 4: File Upload & RAG Ingestion
========================================================
POST /upload
  • Accepts: PDF, TXT, images (PNG/JPG)
  • Extracts text with pypdf (PDF) or direct decode (TXT)
  • Chunks text using custom splitter (no torch needed)
  • Embeds + upserts into Qdrant via vector_db module
  • Stores file metadata in SQL
  • Returns: {file_id, filename, chunks_stored}
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.sql_db import get_db
from app.database.vector_db import upsert_chunks
from app.models.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Supported MIME types ────────────────────────────────────────────────────
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "text/plain":      ".txt",
    "image/png":       ".png",
    "image/jpeg":      ".jpg",
}

MAX_BYTES = settings.max_upload_mb * 1024 * 1024


# ── Text chunker (no langchain-text-splitters required) ───────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Simple but effective character-based chunker with sentence-boundary awareness.
    Falls back to word splitting — no external dependencies.
    """
    chunk_size = chunk_size or settings.chunk_size * 4   # ~800 tokens → ~3200 chars
    overlap    = overlap    or settings.chunk_overlap * 4

    if not text.strip():
        return []

    # Split on sentence boundaries first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n\n", " \n\n "))
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last ~20% of sentences
            keep = max(1, len(current) // 5)
            current = current[-keep:]
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF byte string using pypdf."""
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


# ── Upload endpoint ────────────────────────────────────────────────────────────

@router.post("/pdf", response_model=UploadResponse, summary="Upload a PDF for RAG ingestion")
@router.post("/", response_model=UploadResponse, summary="Upload a file for RAG ingestion")
async def upload_file(
    file:       UploadFile = File(..., description="PDF or TXT file to ingest"),
    session_id: str        = Form(default="", description="Optional session ID to associate the file with"),
    db:         AsyncSession = Depends(get_db),
) -> UploadResponse:
    """
    Upload a PDF or text file. The system will:
    1. Validate file type and size
    2. Extract text (PDF → pypdf, TXT → direct)
    3. Chunk text with overlap
    4. Embed chunks with fastembed (ONNX, ~130MB model)
    5. Upsert into Qdrant vector DB
    6. Return a file_id for use in /chat requests
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Allowed: {list(ALLOWED_TYPES.keys())}",
        )

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_mb} MB.",
        )

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Save to disk ──────────────────────────────────────────────────────────
    file_id  = str(uuid.uuid4())
    ext      = ALLOWED_TYPES[content_type]
    filename = f"{file_id}{ext}"
    save_path = Path(settings.upload_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(raw)
    logger.info("[upload] Saved '%s' → %s (%d bytes)", file.filename, filename, len(raw))

    # ── Extract text ──────────────────────────────────────────────────────────
    if content_type == "application/pdf":
        try:
            text = extract_pdf_text(raw)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF parsing failed: {exc}")
    elif content_type == "text/plain":
        text = raw.decode("utf-8", errors="replace")
    else:
        # Images — cannot extract text without OCR (out of scope for now)
        return UploadResponse(
            file_id=file_id,
            filename=file.filename or filename,
            chunks_stored=0,
            message="Image uploaded and saved. Text extraction from images requires OCR (not configured). Use /chat to reference by file_id.",
        )

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_text(text)
    logger.info("[upload] '%s' → %d chunks", file.filename, len(chunks))

    # ── Embed & upsert to Qdrant ──────────────────────────────────────────────
    metadata = [
        {
            "file_id":    file_id,
            "filename":   file.filename or filename,
            "page_chunk": i,
            "session_id": session_id or "global",
        }
        for i in range(len(chunks))
    ]
    stored = upsert_chunks(chunks, metadata, id_prefix=file_id)

    logger.info("[upload] ✅ Stored %d vectors for file_id=%s", stored, file_id)
    return UploadResponse(
        file_id=file_id,
        filename=file.filename or filename,
        chunks_stored=stored,
        message=f"File ingested successfully. Use file_id='{file_id}' in your /chat requests.",
    )
