"""
app/database/sql_db.py — SQLAlchemy Async Database
====================================================
Manages chat sessions and conversation message history.
Uses SQLite (via aiosqlite) for local dev — swap DATABASE_URL
to postgresql+asyncpg://... for production with zero code change.

Tables:
  sessions  — one row per chat session (id, title, created_at)
  messages  — conversation turns (session_id FK, role, content, ts)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text, event, select
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import settings

logger = logging.getLogger(__name__)

# ── Engine & Session Factory ──────────────────────────────────────────────────
_is_sqlite = settings.database_url.startswith("sqlite")

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,        # logs all SQL in debug mode
    future=True,
    connect_args=(
        {"check_same_thread": False, "timeout": 30}
        if _is_sqlite
        else {}
    ),
)


if _is_sqlite:
    @event.listens_for(engine.sync_engine, "connect")
    def _configure_sqlite(dbapi_connection, _connection_record) -> None:
        """Allow readers and short writes to coexist during streamed requests."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── ORM Base ──────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class Session(Base):
    """One chat session (conversation thread)."""
    __tablename__ = "sessions"

    id         = Column(String(36), primary_key=True)   # UUID string
    title      = Column(String(200), default="New Chat")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    messages   = relationship("Message", back_populates="session",
                              cascade="all, delete-orphan", lazy="select")


class Message(Base):
    """A single conversation turn."""
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(20), nullable=False)    # "user" | "assistant" | "system"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session    = relationship("Session", back_populates="messages")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables if they don't exist (called on app startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("[sql_db] Tables initialised at: %s", settings.database_url)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async DB session per request.
    Usage:
        @router.get("/...")
        async def endpoint(db: AsyncSession = Depends(get_db)):
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Session CRUD ──────────────────────────────────────────────────────────────

async def create_session(db: AsyncSession, session_id: str, title: str = "New Chat") -> Session:
    """Create a new chat session row."""
    s = Session(id=session_id, title=title)
    db.add(s)
    await db.flush()
    logger.debug("[sql_db] Created session: %s", session_id)
    return s


async def get_or_create_session(db: AsyncSession, session_id: str) -> Session:
    """Fetch existing session or create a new one."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    s = result.scalar_one_or_none()
    if s is None:
        s = await create_session(db, session_id)
    return s


async def add_message(
    db: AsyncSession, session_id: str, role: str, content: str
) -> Message:
    """Append a message to a session's history."""
    msg = Message(session_id=session_id, role=role, content=content)
    db.add(msg)
    await db.flush()
    return msg


async def get_history(
    db: AsyncSession, session_id: str, limit: int = 20
) -> list[Message]:
    """
    Fetch the last *limit* messages for a session, oldest first.
    Used to build the LangChain message list for LLM context.
    """
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    msgs = list(reversed(result.scalars().all()))
    logger.debug("[sql_db] Loaded %d messages for session %s", len(msgs), session_id)
    return msgs
