# ui.py is responsible for the frontend interface of my chatbot. 
# It renders HTML pages using Jinja2 templates, displays chat history, creates new chat sessions, 
# sends messages to the LangGraph pipeline, and deletes old chat sessions. 
# It follows Server-Side Rendering (SSR), so no JavaScript is required for basic chat functionality.

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
from fastapi.templating import Jinja2Templates #Loads HTML templates. 
from sqlalchemy import select, desc # Used for database queries 
from sqlalchemy.ext.asyncio import AsyncSession # Used for async database sessions

from app.agents.state import initial_state # Used to initialize the state of the chatbot. 
from app.database.sql_db import (
    AsyncSessionLocal, Session as ChatSession, Message,
    get_db, get_or_create_session, add_message, get_history,
)
from app.graph import get_graph #Returns compiled LangGraph.

logger = logging.getLogger(__name__)
router = APIRouter() #Creates UI Router.
templates = Jinja2Templates(directory="frontend/templates")  #Loads HTML files.


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_all_sessions(db: AsyncSession) -> list[ChatSession]: #Load recent chats.
    """Return all sessions ordered by most recently updated."""
    result = await db.execute(
        select(ChatSession).order_by(desc(ChatSession.updated_at)).limit(30) # read sessionn and newr chats first and latest 30 chats 
    )
    return list(result.scalars().all())

# first msg as title 
async def _update_session_title(db: AsyncSession, session_id: str, first_message: str) -> None:
    """Set the session title to the first ~50 chars of the first user message."""
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    s = result.scalar_one_or_none()
    if s and (s.title == "New Chat" or not s.title): # first msg becomes session title 
        s.title = first_message[:55].strip() # keep first 55 char 
        if len(first_message) > 55:
            s.title += "…"
        s.updated_at = datetime.now(timezone.utc)
        await db.flush() # save changes 

# group chats  today , yesterday ,older 
def _group_sessions_by_day(sessions: list[ChatSession]) -> dict[str, list[ChatSession]]:
    """Group sessions into Today / Yesterday / Older buckets."""
    now = datetime.now(timezone.utc)  # current time 
    groups: dict[str, list] = {"Today": [], "Yesterday": [], "Older": []}
    for s in sessions: # check every chat 
        ts = s.updated_at
        if ts is None:
            groups["Older"].append(s)
            continue
        # Make timezone-aware if naive
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = (now.date() - ts.date()).days # how many old days 
        if delta == 0: # means today        
            groups["Today"].append(s)
        elif delta == 1: # means yesterday 
            groups["Yesterday"].append(s)
        else: # means older 
            groups["Older"].append(s)
    return {k: v for k, v in groups.items() if v}


# ══════════════════════════════════════════════════════════════════════════════
# GET /ui — Landing page (empty new chat)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/", response_class=HTMLResponse) # open /UI 
async def ui_home(request: Request, db: AsyncSession = Depends(get_db)): # get database connection 
    sessions = await _get_all_sessions(db) # get all sessions 
    grouped  = _group_sessions_by_day(sessions) # group sessions by day 
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

@router.get("/chat/{session_id}", response_class=HTMLResponse)# open existing chat session 
async def ui_view_session(session_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id)) # find session 
    active = result.scalar_one_or_none()
    if not active:
        return RedirectResponse("/ui") # if not found backto home 

    messages = await get_history(db, session_id, limit=100) # get previous messages 
    sessions = await _get_all_sessions(db) # get all sessions 
    grouped  = _group_sessions_by_day(sessions) # group sessions by day 

    return templates.TemplateResponse("chat.html", { # open chat html with old conversation 
        "request":          request,
        "active_session":   active,
        "messages":         messages,
        "grouped_sessions": grouped,
        "error":            None,
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /ui/send — Submit a message, run pipeline, redirect
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/send", response_class=RedirectResponse) # run when user click send 
async def ui_send(
    request: Request,
    message:    str = Form(..., min_length=1, max_length=4000), # read msg from html form 
    session_id: str = Form(default=""), # current cha t
    db: AsyncSession = Depends(get_db),
):
    # Create or retrieve session
    sid = session_id.strip() if session_id.strip() else str(uuid.uuid4()) # new chat 
    await get_or_create_session(db, sid) # create session 

    # Load existing history
    history_orm = await get_history(db, sid, limit=20) # Loads previous conversation.
    history = [{"role": m.role, "content": m.content} for m in history_orm] # convert to dict format 

    # Build state & run pipeline
    state = initial_state(  #Creates LangGraph state.
        session_id=sid,
        query=message.strip(),
        history=history,
        uploaded_files=[],
        active_url=None,
    )
    graph = get_graph() #load compiled graph 
    try:
        final_state = await graph.ainvoke(state) # run chat bot 
    except Exception as exc:
        logger.error("[ui/send] Pipeline error: %s", exc, exc_info=True)
        # Save user message anyway, then redirect with error param
        await add_message(db, sid, "user", message.strip())
        return RedirectResponse(f"/ui/chat/{sid}?error=1", status_code=303)

    answer     = final_state.get("final_answer", "Sorry, I could not generate a response.")
    route_used = final_state.get("route_used", "general")

    # Save both turns in sqlite 
    await add_message(db, sid, "user",      message.strip())
    await add_message(db, sid, "assistant", answer)

    # Update session title from first message
    await _update_session_title(db, sid, message.strip())

    return RedirectResponse(f"/ui/chat/{sid}", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# POST /ui/new — Create a brand new session
# ══════════════════════════════════════════════════════════════════════════════
 # redirects to fresh page 
@router.post("/new", response_class=RedirectResponse)
async def ui_new_chat(db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/ui", status_code=303)


# ══════════════════════════════════════════════════════════════════════════════
# POST /ui/delete/{session_id} — Delete a session
# ══════════════════════════════════════════════════════════════════════════════
 # for delete msg s
@router.post("/delete/{session_id}", response_class=RedirectResponse)
async def ui_delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete as sql_delete
    await db.execute(sql_delete(Message).where(Message.session_id == session_id)) # delete msg 
    await db.execute(sql_delete(ChatSession).where(ChatSession.id == session_id))#delete session 
    return RedirectResponse("/ui", status_code=303)


# Browser
#     │
#     ▼
# GET /ui
#     │
#     ▼
# Load Sessions
#     │
#     ▼
# Render chat.html
#     │
#     ▼
# User Types Message
#     │
#     ▼
# POST /ui/send
#     │
#     ▼
# Create LangGraph State
#     │
#     ▼
# Run Graph
#     │
#     ▼
# Save Messages
#     │
#     ▼
# Redirect Back
#     │
#     ▼
# Updated Chat Page

# Jinja2 performs server-side rendering, where HTML is generated on the server before being sent to the browser. 