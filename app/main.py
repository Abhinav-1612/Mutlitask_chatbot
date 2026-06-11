"""
app/main.py — FastAPI Application (Omni-Agent)
===============================================
Entry point. Registers:
  - Lifespan (DB init, Qdrant + graph pre-warm on startup)
  - CORS middleware
  - API routers: /chat, /upload
  - Health, info, root endpoints
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database.sql_db import init_db
from app.models.schemas import HealthResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  🤖  Omni-Agent starting up  (Python 3.13 / Groq)")
    logger.info("=" * 60)

    # Ensure data directories
    for path in [settings.upload_dir, settings.fastembed_cache_dir]:
        os.makedirs(path, exist_ok=True)
    logger.info("[startup] Data directories verified.")

    # Initialise SQL tables (creates omni_agent.db if not exists)
    await init_db()
    logger.info("[startup] ✅ SQLite DB ready.")

    # Pre-compile LangGraph (avoids cold-start on first request)
    from app.graph import compile_graph
    compile_graph()
    logger.info("[startup] ✅ LangGraph pipeline compiled.")

    logger.info("[startup] 🚀 API live at http://%s:%s/docs", settings.app_host, settings.app_port)
    yield

    logger.info("[shutdown] 👋 Omni-Agent shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🤖 Omni-Agent — Universal Multi-Agent Chatbot",
    description=(
        "Production-grade multi-agent chatbot with intelligent routing.\n\n"
        "**Agents**: General Chat • RAG (PDF/URL) • Web Search • Finance/Sports\n\n"
        "**Stack**: FastAPI · LangGraph · Groq (free) · Qdrant · SQLAlchemy · fastembed"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
from app.api.chat   import router as chat_router
from app.api.upload import router as upload_router

app.include_router(chat_router,   prefix="/chat",   tags=["💬 Chat"])
app.include_router(upload_router, prefix="/upload", tags=["📁 Upload"])




# ── Core endpoints ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["⚙️ System"])
async def health():
    return HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.utcnow())


@app.get("/info", tags=["⚙️ System"])
async def info():
    """Runtime configuration (safe fields)."""
    return {
        "models": {
            "router": settings.router_model,
            "agent":  settings.agent_model,
        },
        "embedding_model": settings.embedding_model,
        "vector_db_index": settings.pinecone_index,
        "sql_db":          settings.database_url,
        "upload_dir":      settings.upload_dir,
    }


# ── Dev entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
