"""
app/models/schemas.py — Pydantic Request/Response Schemas
==========================================================
All API input/output contracts for FastAPI endpoints.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


# ── Enums ─────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    user      = "user"
    assistant = "assistant"
    system    = "system"


class AgentRoute(str, Enum):
    general  = "general"       # Chit-chat / general knowledge
    rag      = "rag"           # Document / URL / research RAG
    web      = "web"           # DuckDuckGo web search
    finance  = "finance"       # Stock prices / sports scores


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session UUID. Omit to start a new session.",
    )
    message: str = Field(..., min_length=1, max_length=4000, description="User's message.")
    file_ids: list[str] = Field(default_factory=list, description="IDs of previously uploaded files.")
    active_url: str | None = Field(None, description="Optional URL for context.")
    user_groq_key: str = Field(default="", description="Optional user-supplied Groq API key.")

    model_config = {"json_schema_extra": {"example": {
        "session_id": "abc123",
        "message": "What is the current Tesla stock price?",
        "user_groq_key": "",
    }}}


class MessageOut(BaseModel):
    role:       MessageRole
    content:    str
    created_at: datetime | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply:      str
    route_used: AgentRoute
    sources:    list[dict[str, Any]] = Field(default_factory=list)
    history:    list[MessageOut]     = Field(default_factory=list)


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_id:       str
    filename:      str
    chunks_stored: int
    message:       str


# ── Session ───────────────────────────────────────────────────────────────────

class SessionOut(BaseModel):
    id:         str
    title:      str
    created_at: datetime
    updated_at: datetime


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:    str = "healthy"
    version:   str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
