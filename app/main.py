"""
app/main.py — FastAPI Application (Omni-Agent)
===============================================
Entry point. Registers:
  - Lifespan (DB init, graph pre-warm on startup)
  - CORS middleware
  - API routers: /chat, /upload
  - Health, info, root endpoints
"""
from __future__ import annotations

import logging # Logging is a way to record events that happen while your program is running.
import os
from contextlib import asynccontextmanager #It is used to manage resources that need to be initialized before an application starts and cleaned up after it stops.
from datetime import datetime # It is used to work with dates and times.

import uvicorn #It is an ASGI server, which is a standard interface between Python web frameworks and web servers.
from fastapi import FastAPI # FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.
from fastapi.middleware.cors import CORSMiddleware #Cross-Origin Resource Sharing (CORS). It allows web pages from different domains to communicate with each other.
from fastapi.responses import RedirectResponse # It is used to redirect the user to a different page.
from fastapi.staticfiles import StaticFiles # It is used to serve static files.

from app.config import settings # It is used to store the application settings.
from app.database.sql_db import init_db # It is used to initialize the database.
from app.models.schemas import HealthResponse # It is used to define the response model for the health endpoint.

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────
# ────────────────────────
# all working which should done before app starts means duein g startup 
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
    await init_db()  # it creatign db if not exist otherwise reading db
    logger.info("[startup] ✅ SQLite DB ready.")

    # Pre-compile LangGraph (avoids cold-start on first request)
    from app.graph import compile_graph #Imports graph compiler.
    compile_graph() # Compiles the graph.  Without thisUser's first requestGraph buildsResponse becomes slow.So graph is built during startup.

    logger.info("[startup] ✅ LangGraph pipeline compiled.")

    logger.info("[startup] 🚀 API live at http://%s:%s/docs", settings.app_host, settings.app_port)
    yield

    logger.info("[shutdown] 👋 Omni-Agent shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🤖 Omni-Agent — Universal Multi-Agent Chatbot",
    description=(
        "Production-grade multi-agent chatbot with intelligent routing.\n\n"
        "**Agents**: General Chat • RAG (PDF/URL) • Web Search • Finance/Sports and farmer mode\n\n"
        "**Stack**: FastAPI · LangGraph · Groq (free) · SQLAlchemy · fastembed"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(  # adds middlwrare means every request from user first goes to middleware then api 
    CORSMiddleware, # allow frontend to call backend
    allow_origins=["*"], # allow all websites alll localhosts 
    allow_credentials=True, # allow all cookies, credentials and sessions
    allow_methods=["*"], # allow all HTTP methods GET, POST, PUT, DELETE, PATCH, OPTIONS, etc
    allow_headers=["*"], # allow all headers
)

# ── Register routers ──────────────────────────────────────────────────────────
from app.api.chat   import router as chat_router # import chat api 
from app.api.upload import router as upload_router # import upload api 

app.include_router(chat_router,   prefix="/chat",   tags=["💬 Chat"]) # include chat api in router 
app.include_router(upload_router, prefix="/upload", tags=["📁 Upload"]) # include upload api in router




# ── Core endpoints ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False) # dont show in swagger ui
async def root():
    return RedirectResponse(url="/docs") #Automatically opens localhost:8000/docs


@app.get("/health", response_model=HealthResponse, tags=["⚙️ System"]) # check server 
async def health():
    return HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.utcnow())

#Returns project configuration.
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
