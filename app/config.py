"""
app/config.py — Centralised Settings (pydantic-settings)
=========================================================
All configuration is read from environment variables / .env file.
Access anywhere via: from app.config import settings
"""
from __future__ import annotations #I use it to improve type hint handling and future compatibility with newer Python versions.

from functools import lru_cache #lru_cache caches the result of a function. Instead of creating the Settings object every time, it creates it once and reuses it.
from pydantic import field_validator #Used to validate or modify values before they are stored.
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(  #This tells Pydantic how it should load the settings.Think of it as giving instructions to BaseSettings.
        env_file=".env",
        env_file_encoding="utf-8",#It tells Python how to read the .env file. UTF-8 is the standard encoding and supports almost every language and special character.
        case_sensitive=False,
        extra="ignore",
    )


# #Without BaseSettings, you would have to use:

# import os
# os.getenv("GROQ_API_KEY")


    # ── Groq LLM ─────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    router_model: str = "qwen/qwen3.6-27b"
    agent_model: str = "openai/gpt-oss-120b"

    # ── External API Keys ────────────────────────────────────────────────────
    rapidapi_key: str = ""       # RapidAPI key for Cricbuzz cricket API (free tier)
    cricapi_key: str = ""        # cricapi.com key (free tier at cricapi.com)

    # ── News & Web Search API Keys ───────────────────────────────────────────
    news_api_key: str = ""       # newsapi.org — 200 req/day free tier
    tavily_api_key: str = ""     # tavily.com — 1000 req/month free tier (news fallback)
    currents_api_key: str = ""   # currentsapi.services
    data_gov_api_key: str = ""   # data.gov.in AGMARKNET API key
    agmarknet_resource_id: str = "9ef84268-d588-465a-a308-a864a43d0070" # Default daily wholesale prices


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

#Only one Settings instance is created during the application's lifetime.

@lru_cache(maxsize=1)   
def get_settings() -> Settings:
    return Settings()


# Convenience singleton
settings = get_settings()
