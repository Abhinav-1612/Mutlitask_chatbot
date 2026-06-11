"""Deterministic routing rules for high-confidence agent intents."""
from __future__ import annotations

import re
from typing import Literal

AgentRoute = Literal["general", "rag", "web", "finance"]


def _contains(query: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns)


def is_weather_query(query: str) -> bool:
    return _contains(
        query,
        (
            r"\bweather\b",
            r"\btemperature\b",
            r"\bforecast\b",
            r"\bhumidity\b",
            r"\bhumid\b",
            r"\bwill it rain\b",
            r"\brain(?:ing|y)?\s+(?:in|at|for)\b",
            r"\bweather conditions?\b",
        ),
    )


def is_news_query(query: str) -> bool:
    return _contains(
        query,
        (
            r"\bnews\b",
            r"\bheadlines?\b",
            r"\bbreaking\b",
            r"\bcurrent events?\b",
            r"\btop stories\b",
            r"\bwhat happened (?:today|recently)\b",
            r"\b(?:latest|recent|today'?s|current)\s+(?:updates?|developments?)\b",
        ),
    )


def _is_stock_query(query: str) -> bool:
    return _contains(
        query,
        (
            r"\bstock (?:price|quote|value|performance)\b",
            r"\bshare (?:price|quote|value)\b",
            r"\bticker\b",
            r"\bmarket cap\b",
            r"\b52[- ]week\b",
            r"\b(?:price|quote) of [A-Z]{1,6}(?:\.[A-Z]{1,3})?\b",
            r"\b[A-Z]{1,6}(?:\.[A-Z]{1,3})? stock\b",
        ),
    )


def _is_live_sports_query(query: str) -> bool:
    has_sport = _contains(
        query,
        (
            r"\bcricket\b",
            r"\bipl\b",
            r"\bfootball\b",
            r"\bsoccer\b",
            r"\bnba\b",
            r"\bnfl\b",
            r"\bnhl\b",
            r"\bmlb\b",
            r"\bmatch\b",
            r"\bgame\b",
        ),
    )
    has_live_intent = _contains(
        query,
        (
            r"\blive\b",
            r"\bscore\b",
            r"\bresult\b",
            r"\bschedule\b",
            r"\bfixture\b",
            r"\bstandings?\b",
            r"\btoday\b",
        ),
    )
    return has_sport and has_live_intent


def _is_document_query(query: str, has_files: bool, has_url: bool) -> bool:
    if _contains(
        query,
        (
            r"\barxiv\b",
            r"\bresearch papers?\b",
            r"\bacademic papers?\b",
            r"\bliterature review\b",
            r"\bscientific stud(?:y|ies)\b",
            r"\bpaper on\b",
            r"\bpaper about\b",
        ),
    ):
        return True

    if has_files and _contains(
        query,
        (
            r"\b(?:uploaded|attached) (?:file|document|pdf)\b",
            r"\bthis (?:file|document|pdf|attachment)\b",
            r"\bthe (?:file|document|pdf|attachment)\b",
            r"\baccording to (?:the|this) (?:file|document|pdf)\b",
            r"\bin (?:the|this) (?:file|document|pdf)\b",
        ),
    ):
        return True

    return has_url and _contains(
        query,
        (
            r"\bthis (?:url|link|page|website)\b",
            r"\bthe (?:url|link|page|website)\b",
            r"\bprovided (?:url|link)\b",
        ),
    )


def _is_realtime_web_query(query: str) -> bool:
    if _contains(
        query,
        (
            r"\bsearch (?:the )?web\b",
            r"\bsearch online\b",
            r"\blook (?:it )?up online\b",
            r"\bon the internet\b",
            r"\bexchange rate\b",
            r"\b(?:gold|silver|fuel|gas|petrol|diesel) price\b",
            r"\b(?:flight|train|order|service) status\b",
            r"\btraffic (?:in|near|to|from)\b",
            r"\bwho (?:is|are) (?:the )?(?:president|prime minister|ceo|governor|mayor)\b",
            r"\b(?:president|prime minister|ceo|governor|mayor) of\b",
        ),
    ):
        return True

    has_freshness = _contains(
        query,
        (
            r"\bcurrent\b",
            r"\blatest\b",
            r"\blive\b",
            r"\btoday\b",
            r"\bright now\b",
            r"\bup[- ]to[- ]date\b",
            r"\bmost recent\b",
        ),
    )
    has_dynamic_subject = _contains(
        query,
        (
            r"\bprice\b",
            r"\bexchange rate\b",
            r"\btraffic\b",
            r"\bschedule\b",
            r"\bavailability\b",
            r"\bstatus\b",
            r"\bpresident\b",
            r"\bprime minister\b",
            r"\bceo\b",
            r"\brelease\b",
            r"\bversion\b",
            r"\bwebsite\b",
        ),
    )
    return has_freshness and has_dynamic_subject


def is_context_followup(query: str) -> bool:
    """Identify short turns that depend on the previous user question."""
    if len(query.split()) > 10:
        return False
    return _contains(
        query,
        (
            r"^\s*(?:what|how) about\b",
            r"^\s*and\b",
            r"\bwhat about there\b",
            r"\bhow about there\b",
            r"\bsame (?:for|question)\b",
            r"\bthat one\b",
        ),
    )


def detect_priority_route(
    query: str,
    *,
    has_files: bool = False,
    has_url: bool = False,
    previous_query: str | None = None,
) -> AgentRoute | None:
    """Return a route only when the intent can be identified confidently."""
    if is_weather_query(query) or is_news_query(query):
        return "web"

    if _is_stock_query(query):
        return "finance"

    if _is_live_sports_query(query):
        query_lower = query.lower()
        return "finance" if "cricket" in query_lower or "ipl" in query_lower else "web"

    if _is_realtime_web_query(query):
        return "web"

    if _is_document_query(query, has_files, has_url):
        return "rag"

    if previous_query and is_context_followup(query):
        return detect_priority_route(
            previous_query,
            has_files=has_files,
            has_url=has_url,
        )

    return None


def choose_route(
    query: str,
    *,
    has_files: bool = False,
    has_url: bool = False,
    previous_query: str | None = None,
) -> AgentRoute:
    """Route explicit tool intents; all other questions are normal LLM chat."""
    priority_route = detect_priority_route(
        query,
        has_files=has_files,
        has_url=has_url,
        previous_query=previous_query,
    )
    if priority_route:
        return priority_route

    # If the user has active context (PDF or URL) and didn't trigger another tool,
    # assume they want to chat about their document.
    if has_files or has_url:
        return "rag"

    return "general"

