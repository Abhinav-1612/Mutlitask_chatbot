"""
app/api/ui.py — Server-side rendered HTML UI
=============================================
Handles the chat UI using Jinja2 templates.
All form submissions are processed here — no JS needed.

Routes:
  GET  /ui                    → Home / new chat
  GET  /ui/chat/{session_id}  → View an existing chat session
  POST /ui/send               → Submit a message, redirect back
  POST /ui/new                → Create new session, redirect
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import initial_state
from app.database.sql_db import (
    AsyncSessionLocal, Session as ChatSession, Message,
    get_db, get_or_create_session, add_message, get_history,
)
from app.graph import get_graph

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_all_sessions(db: AsyncSession) -> list[ChatSession]:
    """Return all sessions ordered by most recently updated."""
    result = await db.execute(
        select(ChatSession).order_by(desc(ChatSession.updated_at)).limit(30)
    )
    return list(result.scalars().all())


async def _update_session_title(db: AsyncSession, session_id: str, first_message: str) -> None:
    """Set the session title to the first ~50 chars of the first user message."""
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    s = result.scalar_one_or_none()
    if s and (s.title == "New Chat" or not s.title):
        s.title = first_message[:55].strip()
        if len(first_message) > 55:
            s.title += "…"
        s.updated_at = datetime.now(timezone.utc)
        await db.flush()


def _group_sessions_by_day(sessions: list[ChatSession]) -> dict[str, list[ChatSession]]:
    """Group sessions into Today / Yesterday / Older buckets."""
    now = datetime.now(timezone.utc)
    groups: dict[str, list] = {"Today": [], "Yesterday": [], "Older": []}
    for s in sessions:
        ts = s.updated_at
        if ts is None:
            groups["Older"].append(s)
            continue
        # Make timezone-aware if naive
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = (now.date() - ts.date()).days
        if delta == 0:
            groups["Today"].append(s)
        elif delta == 1:
            groups["Yesterday"].append(s)
        else:
            groups["Older"].append(s)
    return {k: v for k, v in groups.items() if v}


# ══════════════════════════════════════════════════════════════════════════════
# GET /ui — Landing page (empty new chat)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/", response_class=HTMLResponse)
async def ui_home(request: Request, db: AsyncSession = Depends(get_db)):
    sessions = await _get_all_sessions(db)
    grouped  = _group_sessions_by_day(sessions)
    return templates.TemplateResponse("chat.html", {
        "request":          request,
        "active_session":   None,
        "messages":         [],
        "grouped_sessions": grouped,
        "error":            None,
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /ui/chat/{session_id} — View a specific chat session
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/chat/{session_id}", response_class=HTMLResponse)
async def ui_view_session(session_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    active = result.scalar_one_or_none()
    if not active:
        return RedirectResponse("/ui")

    messages = await get_history(db, session_id, limit=100)
    sessions = await _get_all_sessions(db)
    grouped  = _group_sessions_by_day(sessions)

    return templates.TemplateResponse("chat.html", {
        "request":          request,
        "active_session":   active,
        "messages":         messages,
        "grouped_sessions": grouped,
        "error":            None,
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /ui/send — Submit a message, run pipeline, redirect
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/send", response_class=RedirectResponse)
async def ui_send(
    request: Request,
    message:    str = Form(..., min_length=1, max_length=4000),
    session_id: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    # Create or retrieve session
    sid = session_id.strip() if session_id.strip() else str(uuid.uuid4())
    await get_or_create_session(db, sid)

    # Load existing history
    history_orm = await get_history(db, sid, limit=20)
    history = [{"role": m.role, "content": m.content} for m in history_orm]

    # Build state & run pipeline
    state = initial_state(
        session_id=sid,
        query=message.strip(),
        history=history,
        uploaded_files=[],
        active_url=None,
    )
    graph = get_graph()
    try:
        final_state = await graph.ainvoke(state)
    except Exception as exc:
        logger.error("[ui/send] Pipeline error: %s", exc, exc_info=True)
        # Save user message anyway, then redirect with error param
        await add_message(db, sid, "user", message.strip())
        return RedirectResponse(f"/ui/chat/{sid}?error=1", status_code=303)

    answer     = final_state.get("final_answer", "Sorry, I could not generate a response.")
    route_used = final_state.get("route_used", "general")

    # Save both turns
    await add_message(db, sid, "user",      message.strip())
    await add_message(db, sid, "assistant", answer)

    # Update session title from first message
    await _update_session_title(db, sid, message.strip())

    return RedirectResponse(f"/ui/chat/{sid}", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# POST /ui/new — Create a brand new session
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/new", response_class=RedirectResponse)
async def ui_new_chat(db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/ui", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# POST /ui/delete/{session_id} — Delete a session
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/delete/{session_id}", response_class=RedirectResponse)
async def ui_delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete as sql_delete
    await db.execute(sql_delete(Message).where(Message.session_id == session_id))
    await db.execute(sql_delete(ChatSession).where(ChatSession.id == session_id))
    return RedirectResponse("/ui", status_code=303)
