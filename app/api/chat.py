#it connects FastAPI → LangGraph → Database → Response.
#chat.py contains all chat-related APIs. 
# It receives the user's message, loads chat history, invokes the LangGraph multi-agent workflow, stores the conversation in SQLite, and returns the final response. 
# It also supports streaming responses using Server-Sent Events (SSE).
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

import asyncio #Used for asynchronous programming.
import json #Converts Python objects into JSON.
import logging
import time
import uuid#Generates unique Session IDs.
from datetime import datetime, timezone
from typing import AsyncIterator  #Used in streaming. 

from fastapi import APIRouter, Depends, HTTPException, Query  # api router ceates chat api Instead of putting everything inside main.py  we create differentfiles and import from there.
#depends used for database connection and many more   
from sse_starlette.sse import EventSourceResponse #straem responses instead of waiting 
from sqlalchemy.ext.asyncio import AsyncSession  #Async Database Connection.
 
from app.agents.state import initial_state #Creates LangGraph state.
from app.database.sql_db import ( # dtabase connection for chat history 
    get_db, get_or_create_session, add_message, get_history, Session
)
from sqlalchemy import select, desc 
from app.graph import get_graph#Returns compiled LangGraph.
from app.models.schemas import ( #loading schemas for request and response data 
    ChatRequest, ChatResponse, MessageOut, MessageRole, AgentRoute
)

logger = logging.getLogger(__name__) #create logger 
router = APIRouter() #create chat router for chat apis 


# ── Helpers ───────────────────────────────────────────────────────────────────

def _history_to_dicts(messages) -> list[dict]:  #converting sql databse oRM type objects and converting them to dictionary as langgraph wants dictionary 
    """Convert ORM Message objects to plain dicts for LangGraph state."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _history_to_out(messages) -> list[MessageOut]: # converts database objects to pydantic objects before returning response 
    """Convert ORM Message objects to Pydantic output schema."""
    return [
        MessageOut(role=MessageRole(m.role), content=m.content, created_at=m.created_at)
        for m in messages
    ]

#============================================
# POST /chat — Standard (blocking) endpoint

@router.post( #create post chat api 
    "/",
    response_model=ChatResponse, #response must match to chat response schema
    summary="Send a message and get a complete response",
)
async def chat( # main chat function
    request: ChatRequest, # recieves chat request 
    db: AsyncSession = Depends(get_db), # depends on sql databse connection 
) -> ChatResponse:
    """
    Run the full multi-agent pipeline for a user message.

    - Loads conversation history from SQLite
    - Routes through: Gateway → (Supervisor →) Specialist Agent
    - Saves user + assistant turns to SQLite
    - Returns final answer with sources and route metadata
    """
    session_id = request.session_id or str(uuid.uuid4()) #Uses existing session. if not creates new one
    logger.info("[chat] POST session=%s message='%s...'", session_id, request.message[:40])

    # ── Ensure session exists in SQL ──────────────────────────────────────────
    session = await get_or_create_session(db, session_id) # if not ceates new session 

    # ── Load history ──────────────────────────────────────────────────────────
    history_orm = await get_history(db, session_id, limit=20) #Loads previous messages.
    history     = _history_to_dicts(history_orm) # converts to dict format 
    
    if not history: # means it it first msg we store title of the chat 
        session.title = request.message[:40] + ("..." if len(request.message) > 40 else "") # take first 40 chars as title 
        
    session.updated_at = datetime.now(timezone.utc)
    db.add(session) # add session 
    await db.flush() # saves changes in db 

    # Release any session/title write lock before slow LLM and tool calls.
    await db.commit() # save title 

    start_time = time.time()
    event_queue = asyncio.Queue()
    flow_steps = []

    first_node = "Farmer Node" if request.farmer_mode else "Router Node"
    first_msg = "🧭 Farmer Mode active — routing to Farmer Node" if request.farmer_mode else "🧭 Analyzing query intent → Routing to Omni-Agent Node"
    flow_steps.append({"node": first_node, "message": first_msg})

    # ── Build initial state ────────────────────────────────────────────────────
    state = initial_state( # creates langgraph state 
        session_id=session_id,
        query=request.message,
        history=history,
        uploaded_files=[{"file_id": fid} for fid in request.file_ids],
        active_url=request.active_url,
        user_groq_key=request.user_groq_key or "",
        agent_model=request.agent_model,
        farmer_mode=request.farmer_mode,
        event_queue=event_queue,
    )

    # ── Run LangGraph pipeline ────────────────────────────────────────────────
    graph = get_graph() # Return compiled LangGraph.
    try:
        final_state = await graph.ainvoke(state) # run the workflow 
    except Exception as exc:
        logger.error("[chat] Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {exc}")

    while not event_queue.empty():
        try:
            it = event_queue.get_nowait()
            flow_steps.append({"node": it["node"], "message": it["message"]})
        except Exception:
            break

    response_time = round(time.time() - start_time, 2)
    answer     = final_state.get("final_answer", "I'm sorry, I couldn't generate a response.") # read the answer 
    route_used = final_state.get("route_used", "general") # reads which agent answered 
    sources    = final_state.get("sources", []) # gets sources used by agent

    # ── Save messages to SQLite ──────────────────────────────────────────────────
    await add_message(db, session_id, "user",      request.message) # adds user msg to db 
    await add_message(db, session_id, "assistant", answer) # adds assistant msg to db

    # ── Reload history for response ───────────────────────────────────────────
    updated_history = await get_history(db, session_id, limit=10) # load latest conversation 

    return ChatResponse(
        session_id=session_id,
        reply=answer,
        route_used=AgentRoute(route_used),
        sources=sources,
        history=_history_to_out(updated_history),
        response_time=response_time,
        execution_flow=flow_steps,
    )


# =========================================
# GET /chat/stream — SSE Streaming endpoint
# 

@router.get(
    "/stream", # streaming api 
    summary="Stream agent logs and response via Server-Sent Events",
)
async def chat_stream(
    session_id:    str  = Query(default_factory=lambda: str(uuid.uuid4())),
    message:       str  = Query(..., min_length=1), # user question
    file_ids:      str  = Query(default="", description="Comma-separated file IDs"), # uploaded pofs
    active_url:    str  = Query(default=""),
    user_groq_key: str  = Query(default="", description="Optional user Groq API key"),
    agent_model:   str  = Query(default="openai/gpt-oss-120b", description="Model for the agent"),
    farmer_mode:   bool = Query(default=False, description="Toggle Farmer Mode"), # on or off
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    """
    SSE streaming endpoint. Connect and receive real-time events.

    Event types:
      - ``log``      : node transition message {"node": ..., "message": ...}
      - ``result``   : final answer {"answer": ..., "route": ..., "sources": [...], "response_time": ..., "execution_flow": [...]}
      - ``error``    : pipeline error {"error": ...}
    """
    file_id_list = [f.strip() for f in file_ids.split(",") if f.strip()]

    async def event_generator() -> AsyncIterator[dict]: # creates  SSE  stream
        try:
            session = await get_or_create_session(db, session_id) # if not creates new session 
            history_orm = await get_history(db, session_id, limit=20) #Loads previous messages.
            history     = _history_to_dicts(history_orm) # converts to dict format 
            
            if not history: # if not creates title for chat 
                session.title = message[:40] + ("..." if len(message) > 40 else "")
                
            session.updated_at = datetime.now(timezone.utc)
            db.add(session)
            await db.flush()

            # Never hold a SQLite write transaction while the graph calls APIs.
            await db.commit()

            start_time = time.time()
            event_queue = asyncio.Queue()
            flow_steps = []

            first_node = "Farmer Node" if farmer_mode else "Router Node"
            first_msg = "🧭 Farmer Mode active — routing to Farmer Node" if farmer_mode else "🧭 Analyzing query intent → Routing to Omni-Agent Node"
            flow_steps.append({"node": first_node, "message": first_msg})

            yield { # Immediately send initial router log event
                "event": "log",
                "data": json.dumps({"node": first_node, "message": first_msg}),
            }

            state = initial_state(
                session_id=session_id,
                query=message,
                history=history,
                uploaded_files=[{"file_id": fid} for fid in file_id_list],
                active_url=active_url or None,
                user_groq_key=user_groq_key or "",
                agent_model=agent_model,
                farmer_mode=farmer_mode,
                event_queue=event_queue,
            )

            graph = get_graph()
            SENTINEL = object()

            async def _run_workflow():
                try:
                    res = await graph.ainvoke(state)
                    await event_queue.put((SENTINEL, res, None))
                except Exception as err:
                    await event_queue.put((SENTINEL, None, err))

            asyncio.create_task(_run_workflow())

            complete_state = None
            while True:
                item = await event_queue.get()
                if isinstance(item, tuple) and len(item) == 3 and item[0] is SENTINEL:
                    _, res_state, workflow_err = item
                    if workflow_err:
                        raise workflow_err
                    complete_state = res_state
                    break

                flow_steps.append({"node": item["node"], "message": item["message"]})
                yield {
                    "event": "log",
                    "data": json.dumps({"node": item["node"], "message": item["message"]}),
                }

            response_time = round(time.time() - start_time, 2)

            answer       = complete_state.get("final_answer", "No response generated.")
            route_used   = complete_state.get("route_used", "general")
            sources      = complete_state.get("sources", [])
            total_tokens = complete_state.get("total_tokens", 0)
            tools_called = complete_state.get("tools_called", [])

            # Save to SQL
            await add_message(db, session_id, "user",      message)
            await add_message(db, session_id, "assistant", answer)
            await db.commit()

            yield {
                "event": "result",
                "data": json.dumps({
                    "session_id":     session_id,
                    "answer":         answer,
                    "route":          route_used,
                    "sources":        sources,
                    "total_tokens":   total_tokens,
                    "tools_called":   tools_called,
                    "response_time":  response_time,
                    "execution_flow": flow_steps,
                }),
            }

        except asyncio.CancelledError:
            logger.info("[chat/stream] Client disconnected.")
        except Exception as exc:
            await db.rollback()
            logger.error("[chat/stream] Error: %s", exc, exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


# =============================================
# GET /chat/history/{session_id}
# =============================================

# Returns old messages.
# Useful when reopening chat.
@router.get(
    "/history/{session_id}", # get history
    response_model=list[MessageOut], # response model
    summary="Retrieve conversation history for a session", # summary 
)
async def get_chat_history(
    session_id: str, # session id
    limit: int = Query(default=50, ge=1, le=200), # limit
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    """Return the last N messages for a session."""
    messages = await get_history(db, session_id, limit=limit)
    return _history_to_out(messages)


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /chat/session/{session_id}
# ══════════════════════════════════════════════════════════════════════════════
 # delete entire chat 
@router.delete(
    "/session/{session_id}",
    summary="Clear a session's conversation history",
)
async def clear_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a session and all its associated messages."""
    from sqlalchemy import delete
    from app.database.sql_db import Session
    await db.execute(delete(Session).where(Session.id == session_id))
    logger.info("[chat] Deleted session: %s", session_id)
    return {"message": f"Session '{session_id}' deleted.", "session_id": session_id}


# ══════════════════════════════════════════════════════════════════════════════
# GET /chat/sessions — Recent sessions list for sidebar
# ══════════════════════════════════════════════════════════════════════════════
 # returns recent chats on sidebar 
@router.get(
    "/sessions",
    summary="List recent chat sessions",
)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the most recent *limit* sessions (id + title + updated_at + is_pinned)."""
    result = await db.execute(
        select(Session).order_by(desc(Session.is_pinned), desc(Session.updated_at)).limit(limit)
    )
    sessions = result.scalars().all()
    return [
        {
            "id": s.id,
            "title": s.title or "New Chat",
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "is_pinned": s.is_pinned,
        }
        for s in sessions
    ]

# ══════════════════════════════════════════════════════════════════════════════
# PUT /chat/session/{session_id}/pin — Toggle pin status
# ══════════════════════════════════════════════════════════════════════════════
@router.put(
    "/session/{session_id}/pin",
    summary="Toggle pin status of a session",
)
async def toggle_pin_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Toggle the is_pinned flag for a session."""
    session = await get_or_create_session(db, session_id)
    session.is_pinned = not session.is_pinned
    await db.commit()
    logger.info("[chat] Toggled pin for session: %s to %s", session_id, session.is_pinned)
    return {"message": f"Session pin toggled.", "session_id": session_id, "is_pinned": session.is_pinned}

# User
#    │
#    ▼
# POST /chat
#    │
#    ▼
# Load Session
#    │
#    ▼
# Load History
#    │
#    ▼
# Create LangGraph State
#    │
#    ▼
# Gateway
#    │
# Supervisor
#    │
# Specialized Agent
#    │
#    ▼
# Generate Answer
#    │
#    ▼
# Save Messages
#    │
#    ▼
# Return Response

from fastapi import UploadFile, File
import os
import tempfile
from groq import Groq
from app.config import settings

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Groq Whisper API."""
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    # Save uploaded file to temp file
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        client = Groq(api_key=settings.groq_api_key)
        
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
        
        os.remove(tmp_path)
        return {"text": transcription.text}
        
    except Exception as e:
        logger.error("[chat] Transcribe error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))