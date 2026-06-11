"""
app/database/vector_db.py — Pinecone Vector Database (Cloud)
=============================================================
Manages document embeddings for RAG using Pinecone.
Uses fastembed (ONNX) for embedding — no torch, no GPU, ~130 MB.

Index: omni-agent-docs
Vector size: 384  (BAAI/bge-small-en-v1.5)
Distance: Cosine
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

logger = logging.getLogger(__name__)

# ── Embedder singleton ────────────────────────────────────────────────────────
_embedder: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        logger.info("[vector_db] Loading fastembed model: %s", settings.embedding_model)
        os.makedirs(settings.fastembed_cache_dir, exist_ok=True)
        _embedder = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=settings.fastembed_cache_dir,
        )
        logger.info("[vector_db] Embedding model ready.")
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    model = get_embedder()
    return [v.tolist() for v in model.embed(texts)]


def embed_one(text: str) -> list[float]:
    """Embed a single string."""
    return embed([text])[0]


# ── Pinecone client singleton ─────────────────────────────────────────────────
_pc: Pinecone | None = None
_index: Any | None = None


def get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        api_key = settings.pinecone_api_key
        if not api_key:
            logger.warning("[vector_db] PINECONE_API_KEY is not set.")
        logger.info("[vector_db] Connecting to Pinecone...")
        _pc = Pinecone(api_key=api_key)
        _ensure_index(_pc)
    return _pc


def _ensure_index(pc: Pinecone) -> None:
    global _index
    index_name = settings.pinecone_index
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        logger.info("[vector_db] Creating Pinecone index '%s'...", index_name)
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") # adjust region as needed
        )
        logger.info("[vector_db] Index '%s' created.", index_name)
    else:
        logger.debug("[vector_db] Index '%s' already exists.", index_name)
    
    _index = pc.Index(index_name)


def get_index() -> Any:
    if _index is None:
        get_pinecone()
    return _index


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list[str],
    metadata: list[dict[str, Any]],
    id_prefix: str = "doc",
) -> int:
    """
    Embed and upsert a list of text chunks into Pinecone.

    Args:
        chunks   : Raw text chunks to embed and store.
        metadata : Parallel list of payload dicts (source, page, file_id, etc.)
        id_prefix: Prefix for generating deterministic point IDs.

    Returns:
        Number of points upserted.
    """
    if not chunks:
        return 0

    index = get_index()
    vectors = embed(chunks)

    vectors_to_upsert = [
        {
            "id": f"{id_prefix}_{i}_{abs(hash(chunk[:40])) % (2**31)}",
            "values": vec,
            "metadata": {"content": chunk, **meta},
        }
        for i, (chunk, vec, meta) in enumerate(zip(chunks, vectors, metadata))
    ]

    index.upsert(vectors=vectors_to_upsert)
    logger.info("[vector_db] Upserted %d chunks (prefix=%s).", len(vectors_to_upsert), id_prefix)
    return len(vectors_to_upsert)


def similarity_search(
    query: str,
    top_k: int = 5,
    filter_payload: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search against the vector DB.

    Args:
        query          : Query text to embed and search.
        top_k          : Number of results to return.
        filter_payload : Optional {key: value} to filter by payload field.

    Returns:
        List of {score, content, **metadata} dicts, sorted by relevance.
    """
    index = get_index()
    query_vec = embed_one(query)

    pinecone_filter = None
    if filter_payload:
        key, val = next(iter(filter_payload.items()))
        pinecone_filter = {key: {"$eq": val}}

    response = index.query(
        vector=query_vec,
        top_k=top_k,
        filter=pinecone_filter,
        include_metadata=True,
    )

    results = []
    for match in response.matches:
        if match.score < 0.3:
            continue
        metadata = match.metadata or {}
        content = metadata.pop("content", "")
        results.append({
            "score": match.score,
            "content": content,
            **metadata
        })

    logger.info("[vector_db] similarity_search → %d results for query '%s...'", len(results), query[:40])
    return results
