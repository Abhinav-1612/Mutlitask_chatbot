"""
app/api/chat.py — Step 5: Chat Endpoints
=========================================
POST /chat
  • Load conversation history from SQL
  • Invoke LangGraph pipeline
  • Save user + assistant messages to SQL
  • Return structured JSON response

GET /chat/stream?session_id=...&message=...
  • Same pipeline but streams tokens as SSE events
  • Events: "log" (node transitions), "token" (LLM chunks), "complete" (final)

GET /chat/history/{session_id}
  • Return past messages for a session

DELETE /chat/session/{session_id}
  • Clear a session's history
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import initial_state
from app.database.sql_db import (
    get_db, get_or_create_session, add_message, get_history
)
from app.graph import get_graph
from app.models.schemas import (
    ChatRequest, ChatResponse, MessageOut, MessageRole, AgentRoute
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _history_to_dicts(messages) -> list[dict]:
    """Convert ORM Message objects to plain dicts for LangGraph state."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _history_to_out(messages) -> list[MessageOut]:
    """Convert ORM Message objects to Pydantic output schema."""
    return [
        MessageOut(role=MessageRole(m.role), content=m.content, created_at=m.created_at)
        for m in messages
    ]


# ══════════════════════════════════════════════════════════════════════════════
# POST /chat — Standard (blocking) endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send a message and get a complete response",
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Run the full multi-agent pipeline for a user message.

    - Loads conversation history from SQLite
    - Routes through: Gateway → (Supervisor →) Specialist Agent
    - Saves user + assistant turns to SQLite
    - Returns final answer with sources and route metadata
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("[chat] POST session=%s message='%s...'", session_id, request.message[:40])

    # ── Ensure session exists in SQL ──────────────────────────────────────────
    session = await get_or_create_session(db, session_id)

    # ── Load history ──────────────────────────────────────────────────────────
    history_orm = await get_history(db, session_id, limit=20)
    history     = _history_to_dicts(history_orm)
    
    if not history:
        session.title = request.message[:40] + ("..." if len(request.message) > 40 else "")
        db.add(session)
        await db.flush()

    # Release any session/title write lock before slow LLM and tool calls.
    await db.commit()

    # ── Build initial state ────────────────────────────────────────────────────
    state = initial_state(
        session_id=session_id,
        query=request.message,
        history=history,
        uploaded_files=[{"file_id": fid} for fid in request.file_ids],
        active_url=request.active_url,
    )

    # ── Run LangGraph pipeline ────────────────────────────────────────────────
    graph = get_graph()
    try:
        final_state = await graph.ainvoke(state)
    except Exception as exc:
        logger.error("[chat] Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {exc}")

    answer     = final_state.get("final_answer", "I'm sorry, I couldn't generate a response.")
    route_used = final_state.get("route_used", "general")
    sources    = final_state.get("sources", [])

    # ── Save messages to SQL ──────────────────────────────────────────────────
    await add_message(db, session_id, "user",      request.message)
    await add_message(db, session_id, "assistant", answer)

    # ── Reload history for response ───────────────────────────────────────────
    updated_history = await get_history(db, session_id, limit=10)

    return ChatResponse(
        session_id=session_id,
        reply=answer,
        route_used=AgentRoute(route_used),
        sources=sources,
        history=_history_to_out(updated_history),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /chat/stream — SSE Streaming endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/stream",
    summary="Stream agent logs and response via Server-Sent Events",
)
async def chat_stream(
    session_id: str  = Query(default_factory=lambda: str(uuid.uuid4())),
    message:    str  = Query(..., min_length=1),
    file_ids:   str  = Query(default="", description="Comma-separated file IDs"),
    active_url: str  = Query(default=""),
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """
    SSE streaming endpoint. Connect and receive real-time events.

    Event types:
      - ``log``      : node transition message {"node": ..., "message": ...}
      - ``result``   : final answer {"answer": ..., "route": ..., "sources": [...]}
      - ``error``    : pipeline error {"error": ...}

    Usage (curl):
        curl -N "http://localhost:8000/chat/stream?session_id=abc&message=Hello"
    """
    file_id_list = [f.strip() for f in file_ids.split(",") if f.strip()]

    async def event_generator() -> AsyncIterator[dict]:
        try:
            session = await get_or_create_session(db, session_id)
            history_orm = await get_history(db, session_id, limit=20)
            history     = _history_to_dicts(history_orm)
            
            if not history:
                session.title = message[:40] + ("..." if len(message) > 40 else "")
                db.add(session)
                await db.flush()

            # Never hold a SQLite write transaction while the graph calls APIs.
            await db.commit()

            state = initial_state(
                session_id=session_id,
                query=message,
                history=history,
                uploaded_files=[{"file_id": fid} for fid in file_id_list],
                active_url=active_url or None,
            )

            yield {
                "event": "log",
                "data": json.dumps({"node": "system", "message": f"🤖 Processing: '{message[:50]}'"}),
            }

            graph       = get_graph()
            seen_logs   : set[str] = set()
            complete_state: dict = {}

            # Stream full state at each step
            async for chunk in graph.astream(state, stream_mode="values"):
                complete_state = chunk

                # Emit new log lines
                for log_line in chunk.get("logs", []):
                    if log_line not in seen_logs:
                        seen_logs.add(log_line)
                        # Try to extract node name from log format "[HH:MM:SS][node_name]"
                        node = "system"
                        if "][" in log_line:
                            try:
                                node = log_line.split("][")[1].split("]")[0]
                            except Exception:
                                pass
                        yield {
                            "event": "log",
                            "data": json.dumps({"node": node, "message": log_line}),
                        }

                # Emit route decision log once we have a route (if not 'general')
                if chunk.get("next_node") and chunk.get("next_node") != "general" and "route_logged" not in seen_logs:
                    seen_logs.add("route_logged")
                    yield {
                        "event": "log",
                        "data": json.dumps({
                            "node": "router",
                            "message": f"🔀 Routing to: {chunk['next_node'].upper()}",
                        }),
                    }
                    
                    # Emit tool notification
                    route = chunk.get("route_used", chunk.get("next_node"))
                    tool_labels = {
                        "web": "🌐 Web Search Tool",
                        "finance": "📈 Finance Tool",
                        "rag": "📚 Document Retrieval Tool",
                    }
                    if route in tool_labels:
                        yield {
                            "event": "log",
                            "data": json.dumps({
                                "node": "router",
                                "message": f"⚙️ Using: {tool_labels[route]}",
                            }),
                        }

                await asyncio.sleep(0.02)

            answer     = complete_state.get("final_answer", "No response generated.")
            route_used = complete_state.get("route_used", "general")
            sources    = complete_state.get("sources", [])

            # Save to SQL
            await add_message(db, session_id, "user",      message)
            await add_message(db, session_id, "assistant", answer)
            await db.commit()

            yield {
                "event": "result",
                "data": json.dumps({
                    "session_id": session_id,
                    "answer":     answer,
                    "route":      route_used,
                    "sources":    sources,
                }),
            }

        except asyncio.CancelledError:
            logger.info("[chat/stream] Client disconnected.")
        except Exception as exc:
            await db.rollback()
            logger.error("[chat/stream] Error: %s", exc, exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


# ══════════════════════════════════════════════════════════════════════════════
# GET /chat/history/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/history/{session_id}",
    response_model=list[MessageOut],
    summary="Retrieve conversation history for a session",
)
async def get_chat_history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    """Return the last N messages for a session."""
    messages = await get_history(db, session_id, limit=limit)
    return _history_to_out(messages)


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /chat/session/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

@router.delete(
    "/session/{session_id}",
    summary="Clear a session's conversation history",
)
async def clear_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete all messages for a session (keeps the session row)."""
    from sqlalchemy import delete
    from app.database.sql_db import Message
    await db.execute(delete(Message).where(Message.session_id == session_id))
    logger.info("[chat] Cleared history for session: %s", session_id)
    return {"message": f"Session '{session_id}' history cleared.", "session_id": session_id}
