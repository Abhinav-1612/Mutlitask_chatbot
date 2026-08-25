"""
app/agents/state.py — LangGraph UniversalAgentState
=====================================================
The single shared TypedDict propagated through every node.
Annotated reducers handle parallel branch merging safely.
"""

#This file defines the shared state of my LangGraph workflow. 
# Every node reads and updates this state while processing the user's request.


from __future__ import annotations

import operator 
from typing import Annotated, TypedDict, Sequence


class UniversalAgentState(TypedDict): # this stores everything needed by chatbot 
    """Shared state for the entire LangGraph pipeline."""

    # ── Conversation ──────────────────────────────────────────────────────────
    session_id: str # store chat session id 
    """Active chat session UUID (matches SQL sessions.id)."""

    query: str  # store current user query 
    """The raw user message for this turn."""

    farmer_mode: bool
    """Toggle to route directly to the specialized Farmer node."""

    messages: Annotated[list[dict], operator.add]  # history of msg added 
    """
    Full conversation history as list of {"role": ..., "content": ...} dicts.
    Uses operator.add reducer so nodes append without overwriting.
    """

    # ── Routing ───────────────────────────────────────────────────────────────
    next_node: str # store gateway decision 
    """
    Routing decision set by gateway_router or supervisor_node.
    Values: "general" | "rag" | "web" | "finance"
    """

    route_used: str # store which agent actually answers 
    """Which specialist ultimately handled the query (for response metadata)."""

    # ── Context inputs ────────────────────────────────────────────────────────
    uploaded_files: list[dict] # store uploaded pdf and doc files 
    """List of file metadata dicts {file_id, filename, mime_type} for RAG."""

    active_url: str | None # store url if provided 
    """Optional URL the user provided for context ingestion."""

    # ── Retrieved context ─────────────────────────────────────────────────────
    rag_context: Annotated[list[str], operator.add] # retriece chunks from vdbase 
    """Text chunks retrieved from Qdrant for RAG grounding."""

    sources: Annotated[list[dict], operator.add] # store sources used to generate ans 
    """Source documents/URLs used to generate the answer."""

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str # final llm response 
    """The assembled Markdown response sent back to the user."""

    # ── Observability ─────────────────────────────────────────────────────────
    logs: Annotated[list[str], operator.add]
    """Node transition log lines streamed via SSE."""

    user_groq_key: str # user own api key 
    """Optional Groq API key supplied by the user — overrides server key if set."""

    agent_model: str
    """The specific Groq LLM model to use for the agent."""


def initial_state( # create fresh state before graph starts 
    session_id: str,
    query: str,
    history: list[dict] | None = None,
    uploaded_files: list[dict] | None = None,
    active_url: str | None = None,
    user_groq_key: str = "",
    agent_model: str = "openai/gpt-oss-120b",
    farmer_mode: bool = False,
) -> UniversalAgentState:
    """Build a clean initial state for a new graph invocation."""
    return UniversalAgentState(
        session_id=session_id,
        query=query,
        farmer_mode=farmer_mode,
        messages=history or [],
        next_node="general",
        route_used="general",
        uploaded_files=uploaded_files or [],
        active_url=active_url,
        rag_context=[],
        sources=[],
        final_answer="",
        user_groq_key=user_groq_key or "",
        agent_model=agent_model,
        logs=[f"[graph] Session {session_id} — query: {query[:60]}..."],
    )
 
#  User Message
#       │
#       ▼
# initial_state()
#       │
#       ▼
# UniversalAgentState
#       │
#       ▼
# Gateway
#       │
#       ▼
# Supervisor
#       │
#       ▼
# Agent
#       │
#       ▼
# Update State
#       │
#       ▼
# Return Final Answer

# "This file defines the shared LangGraph state using a TypedDict called UniversalAgentState. It stores the user's query, chat history, routing decisions, uploaded files, retrieved document context, sources, logs, and the final response. Every node in the workflow reads from and updates this shared state, enabling smooth communication between agents."