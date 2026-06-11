"""
app/agents/state.py — LangGraph UniversalAgentState
=====================================================
The single shared TypedDict propagated through every node.
Annotated reducers handle parallel branch merging safely.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict, Sequence


class UniversalAgentState(TypedDict):
    """Shared state for the entire LangGraph pipeline."""

    # ── Conversation ──────────────────────────────────────────────────────────
    session_id: str
    """Active chat session UUID (matches SQL sessions.id)."""

    query: str
    """The raw user message for this turn."""

    messages: Annotated[list[dict], operator.add]
    """
    Full conversation history as list of {"role": ..., "content": ...} dicts.
    Uses operator.add reducer so nodes append without overwriting.
    """

    # ── Routing ───────────────────────────────────────────────────────────────
    next_node: str
    """
    Routing decision set by gateway_router or supervisor_node.
    Values: "general" | "rag" | "web" | "finance"
    """

    route_used: str
    """Which specialist ultimately handled the query (for response metadata)."""

    # ── Context inputs ────────────────────────────────────────────────────────
    uploaded_files: list[dict]
    """List of file metadata dicts {file_id, filename, mime_type} for RAG."""

    active_url: str | None
    """Optional URL the user provided for context ingestion."""

    # ── Retrieved context ─────────────────────────────────────────────────────
    rag_context: Annotated[list[str], operator.add]
    """Text chunks retrieved from Qdrant for RAG grounding."""

    sources: Annotated[list[dict], operator.add]
    """Source documents/URLs used to generate the answer."""

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str
    """The assembled Markdown response sent back to the user."""

    # ── Observability ─────────────────────────────────────────────────────────
    logs: Annotated[list[str], operator.add]
    """Node transition log lines streamed via SSE."""


def initial_state(
    session_id: str,
    query: str,
    history: list[dict] | None = None,
    uploaded_files: list[dict] | None = None,
    active_url: str | None = None,
) -> UniversalAgentState:
    """Build a clean initial state for a new graph invocation."""
    return UniversalAgentState(
        session_id=session_id,
        query=query,
        messages=history or [],
        next_node="general",
        route_used="general",
        uploaded_files=uploaded_files or [],
        active_url=active_url,
        rag_context=[],
        sources=[],
        final_answer="",
        logs=[f"[graph] Session {session_id} — query: {query[:60]}..."],
    )
