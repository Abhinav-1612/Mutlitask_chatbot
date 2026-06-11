"""
app/config.py — Centralised Settings (pydantic-settings)
=========================================================
All configuration is read from environment variables / .env file.
Access anywhere via: from app.config import settings
"""
from __future__ import annotations

from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq LLM ─────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    router_model: str = "llama-3.1-8b-instant"
    agent_model: str = "llama-3.3-70b-versatile"

    # ── SQL Database ──────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/omni_agent.db"

    # ── Pinecone Vector DB ────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index: str = "omni-agent-docs-384"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    fastembed_cache_dir: str = "./.cache/fastembed"

    # ── File Uploads ──────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50

    # ── RAG Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_mode(cls, value):
        """Accept common deployment-mode strings from hosted environments."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"dev", "development"}:
                return True
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience singleton
settings = get_settings()
