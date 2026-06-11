"""
app/tools/arxiv_tool.py — ArXiv Academic Paper Search
======================================================
Uses the official arxiv Python library (free, no API key).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import arxiv

logger = logging.getLogger(__name__)


def _search_arxiv_sync(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Blocking arXiv search."""
    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=1.0, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append({
                "title":    paper.title.strip(),
                "authors":  [a.name for a in paper.authors[:3]],
                "year":     paper.published.year if paper.published else None,
                "abstract": paper.summary.strip()[:500],
                "pdf_url":  paper.pdf_url,
                "entry_id": paper.entry_id.split("/")[-1],
            })
        return results
    except Exception as exc:
        logger.error("[arxiv] Search error: %s", exc)
        return []


async def search_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Async arXiv search."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _search_arxiv_sync, query, max_results)
    logger.info("[arxiv] '%s' → %d papers", query, len(results))
    return results


def format_arxiv_results(results: list[dict]) -> str:
    """Format arXiv results as markdown."""
    if not results:
        return "No arXiv papers found for this query."
    lines = ["## 📄 arXiv Research Papers\n"]
    for i, p in enumerate(results, 1):
        authors = ", ".join(p["authors"])
        lines.append(f"**[{i}] {p['title']}** ({p['year']})")
        lines.append(f"*{authors}*")
        lines.append(f"[PDF]({p['pdf_url']}) | ID: {p['entry_id']}")
        lines.append(f"> {p['abstract'][:300]}...\n")
    return "\n".join(lines)
