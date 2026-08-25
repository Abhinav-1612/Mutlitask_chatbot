"""
app/models/schemas.py — Pydantic Request/Response Schemas
==========================================================
All API input/output contracts for FastAPI endpoints.
"""

# This file contains all the Pydantic models used by FastAPI. 
# These models validate incoming requests and define the structure of API responses, 
# ensuring that data sent between the frontend and backend is consistent and 
# correctly formatted.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field #Used to create request and response models.
import uuid


# ── Enums ─────────────────────────────────────────────────────────────────────
# It defines who sent the message. only the followed are allowed not any other 
class MessageRole(str, Enum):
    user      = "user"   #The person messaging
    assistant = "assistant" # The AI model
    system    = "system"   # The system prompt defining the AI's behavior

# Stores which agent handled the user's query.
class AgentRoute(str, Enum):
    general  = "general"       # Chit-chat / general knowledge
    rag      = "rag"           # Document / URL / research RAG
    web      = "web"           # DuckDuckGo web search
    finance  = "finance"       # Stock prices / sports scores


# ── Chat ──────────────────────────────────────────────────────────────────────

# whenever frontend sends data fast pai converts that to chatrequest object.
# this is the request model for the chat endpoint 
# It defines the input structure for a chat message.
class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session UUID. Omit to start a new session.",
    )
    message: str = Field(..., min_length=1, max_length=4000, description="User's message.")
    file_ids: list[str] = Field(default_factory=list, description="IDs of previously uploaded files.")
    active_url: str | None = Field(None, description="Optional URL for context.")
    user_groq_key: str = Field(default="", description="Optional user-supplied Groq API key.")
    agent_model: str = Field(default="openai/gpt-oss-120b", description="Model to use for the agent.")
    farmer_mode: bool = Field(default=False, description="Whether to route to the specialized farmer agent.")

    model_config = {"json_schema_extra": {"example": {
        "session_id": "abc123",
        "message": "What is the current Tesla stock price?",
        "user_groq_key": "",
        "agent_model": "openai/gpt-oss-120b",
        "farmer_mode": False,
    }}}

# Represents one message returned to frontend. used while showing chat history 
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
# response model for the upload endpoint. basically when file is uploaded we send back the file id , filename , chunks stored and a message
class UploadResponse(BaseModel):
    file_id:       str
    filename:      str
    chunks_stored: int
    message:       str


# ── Session ───────────────────────────────────────────────────────────────────
# it is used to show the session history to the user.
class SessionOut(BaseModel):
    id:         str
    title:      str
    created_at: datetime
    updated_at: datetime


# ── Health ────────────────────────────────────────────────────────────────────
# return health endpoint 
class HealthResponse(BaseModel):
    status:    str = "healthy"
    version:   str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
