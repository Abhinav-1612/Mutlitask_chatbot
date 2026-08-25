
#upload.py handles document uploads for my RAG chatbot. 
# It accepts PDF, TXT, and image files, extracts text, splits it into chunks, 
# generates embeddings, stores them in the vector database, 
# and returns a file ID that can later be used during chat.

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
import uuid #Every uploaded file gets a unique ID. even if 2 users upload file in different chats their id will be different 
from pathlib import Path #Used to work with file paths.
from typing import Annotated # Used for type hinting and indicates that a variable is expected to be of a certain type.

from fastapi import APIRouter, Depends, File , Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings  #Reads Upload folder Max upload size from config.py.
from app.database.sql_db import get_db #Creates database connection.
from app.database.vector_db import upsert_chunks # Stores vectors in Pinecone (serverless vector DB).
from app.models.schemas import UploadResponse # Provides data structure for upload responses.

logger = logging.getLogger(__name__) #Creates a logger.
router = APIRouter() # Creates router for chat related apis.

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

    if not text.strip(): # if file empty then retunrn 
        return []

    # Split on sentence boundaries first
    import re #used for pattern matching and manipulating strings in python 
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n\n", " \n\n ")) #Splits the text into sentences.
    chunks, current, current_len = [], [], 0 
    for sent in sentences: # read every sentence 
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current: # if current chunk full save it 
            chunks.append(" ".join(current))
            # Overlap: keep last ~20% of sentences
            keep = max(1, len(current) // 5)
            current = current[-keep:] # keep last few sentences 
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]  # return chunks 
 

# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF byte string using pypdf."""
    import io
    from pypdf import PdfReader # for reading pdf 
    reader = PdfReader(io.BytesIO(file_bytes))
    pages  = []
    for i, page in enumerate(reader.pages): # read every pages 
        text = page.extract_text() or "" # extratct text 
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages) # return complete document 


# ── Upload endpoint ────────────────────────────────────────────────────────────

@router.post("/pdf", response_model=UploadResponse, summary="Upload a PDF for RAG ingestion")
@router.post("/", response_model=UploadResponse, summary="Upload a file for RAG ingestion")
async def upload_file(
    file:       UploadFile = File(..., description="PDF or TXT file to ingest"), # receive uploaded file 
    session_id: str        = Form(default="", description="Optional session ID to associate the file with"), # for future use 
    db:         AsyncSession = Depends(get_db), # Database connection 
) -> UploadResponse:
    """
    Upload a PDF or text file. The system will:
    1. Validate file type and size
    2. Extract text (PDF → pypdf, TXT → direct)
    3. Chunk text with overlap
    4. Embed chunks with fastembed (ONNX, ~130MB model)
    5. Upsert into Pinecone vector DB
    6. Return a file_id for use in /chat requests
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    content_type = file.content_type or ""  # reading file type 
    if content_type not in ALLOWED_TYPES: #checking if file type is allowed 
        raise HTTPException(
            status_code=415, # reject unsupported file
            detail=f"Unsupported file type: {content_type}. Allowed: {list(ALLOWED_TYPES.keys())}",
        )

    raw = await file.read() # read file bytes 
    if len(raw) > MAX_BYTES: # if length too large reject it 
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_mb} MB.",
        )

    if len(raw) == 0: # if empty then also reject 
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Save to disk ──────────────────────────────────────────────────────────
    file_id  = str(uuid.uuid4())# generate unique id 
    ext      = ALLOWED_TYPES[content_type]
    filename = f"{file_id}{ext}" # create file name 
    save_path = Path(settings.upload_dir) / filename # where file saved 
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(raw) # actually saves file to disc
    logger.info("[upload] Saved '%s' → %s (%d bytes)", file.filename, filename, len(raw))

    # ── Extract text ──────────────────────────────────────────────────────────
    if content_type == "application/pdf":
        try:
            text = extract_pdf_text(raw) # if pdf then extracting text 
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF parsing failed: {exc}")
    elif content_type == "text/plain":  # else read direct 
        text = raw.decode("utf-8", errors="replace")
    else: # for images  
        # Images — cannot extract text without OCR (out of scope for now)
        return UploadResponse(
            file_id=file_id,
            filename=file.filename or filename,
            chunks_stored=0,
            message="Image uploaded and saved. Text extraction from images requires OCR (not configured). Use /chat to reference by file_id.",
        )

    if not text.strip(): # if not read text then return eroor 
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_text(text) #120 chunks 
    logger.info("[upload] '%s' → %d chunks", file.filename, len(chunks))

    # ── Embed & upsert to Pineconne ──────────────────────────────────────────────
    metadata = [ # everything stored to metadata 
        {
            "file_id":    file_id,
            "filename":   file.filename or filename,
            "page_chunk": i,
            "session_id": session_id or "global",
        }
        for i in range(len(chunks))
    ]
    stored = upsert_chunks(chunks, metadata, id_prefix=file_id) # then after embedding all stored to pineconne

    logger.info("[upload] ✅ Stored %d vectors for file_id=%s", stored, file_id)
    return UploadResponse( # return all things  -
        file_id=file_id,
        filename=file.filename or filename,
        chunks_stored=stored,
        message=f"File ingested successfully. Use file_id='{file_id}' in your /chat requests.",
    )

    
# User Uploads PDF
#         │
#         ▼
# Validate File
#         │
#         ▼
# Save File
#         │
#         ▼
# Extract Text
#         │
#         ▼
# Split into Chunks
#         │
#         ▼
# Generate Embeddings
#         │
#         ▼
# Store in Qdrant
#         │
#         ▼
# Return File ID