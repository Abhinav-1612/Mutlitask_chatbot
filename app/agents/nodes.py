"""
app/agents/nodes.py — All LangGraph Agent Nodes
=================================================
Implements 6 nodes:

  gateway_router   — Zero-shot intent classifier (Llama 3.1-8b)
                     Routes: general | rag | web | finance

  supervisor_node  — Re-validates and refines routing decision
                     (safety net for ambiguous cases)

  general_node     — Direct conversational LLM (Llama 3.3-70b)
                     Chit-chat, general knowledge, no tools

  rag_node         — Qdrant similarity search → grounded LLM answer
                     For uploaded PDFs, URLs, academic research

  web_node         — DuckDuckGo search → LLM synthesis
                     For real-time internet data

  finance_node     — yfinance + cricket API → structured answer
                     For stock prices and live sports scores
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from app.agents.state import UniversalAgentState
from app.agents.routing import (
    choose_route,
    is_news_query,
    is_weather_query,
    is_instagram_query,
    is_cricket_score_query,
)
from app.config import settings
from app.database.vector_db import similarity_search
from app.tools.search import (
    web_search,
    news_search,
    format_search_results,
    format_news_results,
    get_weather,
    format_weather_result,
    get_instagram_news,
    format_instagram_results,
)
from app.tools.finance import (
    get_stock_price, format_stock_result,
    get_cricket_scores, format_cricket_result,
)
from app.tools.arxiv_tool import search_arxiv, format_arxiv_results
from app.tools.agriculture import get_mandi_prices, format_mandi_prices

logger = logging.getLogger(__name__)

# ── LLM factory — supports per-user Groq API keys ────────────────────────────
_llm_cache: dict[str, tuple] = {}   # key → (llm_fast, llm_smart)

def _get_llms(user_key: str = "") -> tuple:
    """
    Return (llm_fast, llm_smart) for the given API key.
    Instances are cached per key so we don’t recreate on every token.
    Falls back to the server’s GROQ_API_KEY if user_key is empty.
    """
    api_key = (user_key.strip() or settings.groq_api_key)
    if api_key not in _llm_cache:
        _llm_cache[api_key] = (
            ChatGroq(
                model=settings.router_model,
                api_key=api_key,
                temperature=0.0,
                max_tokens=512,
            ),
            ChatGroq(
                model=settings.agent_model,
                api_key=api_key,
                temperature=0.3,
                max_tokens=2048,
            ),
        )
    return _llm_cache[api_key]

# ── Tools for general_node ────────────────────────────────────────────────────
@tool
async def tool_get_weather(location: str) -> str:
    """Get the current weather and 3-day forecast for any city or location."""
    data = await get_weather(location)
    return format_weather_result(data)


@tool
async def tool_web_search(query: str) -> str:
    """
    Search the web for current news, recent events, or any real-time information.
    Use this for: latest news, current events, trending topics, live scores, recent data.
    Do NOT use for: general knowledge, math, history, definitions.
    """
    results = await web_search(query, max_results=6, news="news" in query.lower())
    if not results:
        return "No results found."
    return format_search_results(results)


# ── Helper ─────────────────────────────────────────────────────────────────────
def _ts(node: str, msg: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}][{node}] {msg}"
    logger.info(entry)
    return entry


def _build_history_context(messages: list[dict], limit: int = 6) -> str:
    """Format last N messages as a conversation block for LLM context."""
    recent = messages[-limit:] if len(messages) > limit else messages
    if not recent:
        return "No prior conversation."
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)


def _last_user_message(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content")
    return None


def _extract_weather_location(query: str) -> str:
    """Extract location name from a weather query robustly."""
    # Handle follow-ups: "what about prayagraj"
    followup = re.search(
        r"^\s*(?:what|how)\s+about\s+(.+?)[?.!]*$",
        query,
        flags=re.IGNORECASE,
    )
    if followup:
        return followup.group(1).strip(" ,?.!")

    # Match: "weather of/in/for/at <location>"
    match = re.search(
        r"\b(?:of|in|for|at)\s+([A-Za-z][\w\s,]+?)(?:\s+(?:today|tomorrow|tonight|right\s*now|now|at\s*current|current|currently))?[?.!]*$",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" ,?.!")

    # Final fallback: strip all weather-related words, return what's left
    cleaned = re.sub(
        r"\b(?:what(?:'s| is)?|how(?:'s| is)?|the|current|currently|at current|weather|temperature|forecast|"
        r"humidity|conditions?|today|tomorrow|tonight|right\s*now|now|please|tell|me|of|is|get|india)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,?.!")



# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Gateway Router (Zero-Shot Guardrail)
# ══════════════════════════════════════════════════════════════════════════════

async def gateway_router(state: UniversalAgentState) -> dict:
    """
    Route explicit tool intents deterministically.
    Questions without a clear tool requirement go directly to general chat.
    """
    query = state["query"]
    previous_query = _last_user_message(state.get("messages", []))
    has_files = bool(state.get("uploaded_files"))
    has_url   = bool(state.get("active_url"))
    logs      = [_ts("gateway_router", f"Classifying query: '{query[:60]}...'")]

    route = choose_route(
        query,
        has_files=has_files,
        has_url=has_url,
        previous_query=previous_query,
    )
    reason = "explicit tool intent" if route != "general" else "no tool required"
    logs.append(_ts("gateway_router", f"Route decision: {route.upper()} ({reason})"))
    return {"next_node": route, "route_used": route, "logs": logs}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Supervisor Agent (Central Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

async def supervisor_node(state: UniversalAgentState) -> dict:
    """
    Secondary validation layer. Re-evaluates routing with broader context.
    Can refine: web → rag (if docs are relevant), finance → web (no ticker found).
    """
    query  = state["query"]
    route  = state["next_node"]
    logs   = [_ts("supervisor_node", f"Validating route '{route}' for: '{query[:50]}'")]
    llm_fast, _ = _get_llms(state.get("user_groq_key", ""))

    # For finance queries, extract ticker symbols for yfinance
    if route == "finance":
        ticker_prompt = (
            f"Extract the stock ticker symbol from this query (e.g. AAPL, TSLA, RELIANCE.NS). "
            f"Reply with ONLY the ticker or 'NONE' if this is about sports/scores.\n\nQuery: {query}"
        )
        resp = await llm_fast.ainvoke([HumanMessage(content=ticker_prompt)])
        ticker = resp.content.strip().upper()
        if ticker != "NONE" and len(ticker) <= 12:
            logs.append(_ts("supervisor_node", f"Extracted ticker: {ticker}"))
            return {
                "logs": logs,
                "sources": [{"type": "ticker", "value": ticker}],
            }

    logs.append(_ts("supervisor_node", f"Route confirmed: {route.upper()}"))
    return {"logs": logs}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — General Conversation Node
# ══════════════════════════════════════════════════════════════════════════════

async def general_node(state: UniversalAgentState) -> dict:
    """
    Conversational LLM with optional tool access.
    The model decides whether to use tools (weather/search) or answer directly.
    Basic/timeless questions are answered directly; real-time queries use tools.
    """
    query = state["query"]
    logs  = [_ts("general_node", "Generating response...")]
    _, llm_smart = _get_llms(state.get("user_groq_key", ""))

    history_ctx = _build_history_context(state.get("messages", []))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    system = (
        "You are Omni-Agent, a helpful, friendly, and knowledgeable AI assistant. "
        f"Today's date and time (UTC): {now}. "
        "TOOL USAGE RULES — follow these strictly:\n"
        "  • Call tool_get_weather ONLY when the user asks about current/live weather or temperature for a specific place.\n"
        "  • Call tool_web_search ONLY when the user asks about: latest news, current events, recent scores, live data, or anything that changes day-to-day.\n"
        "  • Do NOT call any tool for: greetings, math, definitions, history, general knowledge, or questions you already know the answer to.\n"
        "  • If the user's question is about news for a specific place (e.g. 'news in Delhi'), search for '<place> latest news today'.\n"
        "Respond clearly and concisely with Markdown formatting where helpful. "
        "Never make up facts — if uncertain, say so."
    )

    _tools = [tool_get_weather, tool_web_search]
    llm_with_tools = llm_smart.bind_tools(_tools)

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Conversation history:\n{history_ctx}\n\nUser: {query}"),
    ]

    response = await llm_with_tools.ainvoke(messages)

    # ── Handle tool calls ──────────────────────────────────────────────────────
    max_rounds = 3
    for _ in range(max_rounds):
        if not response.tool_calls:
            break

        messages.append(response)
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            logs.append(_ts("general_node", f"Tool call: {name}({args})"))
            try:
                if name == "tool_get_weather":
                    result = await tool_get_weather.ainvoke(args)
                elif name == "tool_web_search":
                    result = await tool_web_search.ainvoke(args)
                else:
                    result = f"Unknown tool: {name}"
            except Exception as exc:
                result = f"Tool error: {exc}"
                logger.error("[general_node] Tool %s failed: %s", name, exc)

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        response = await llm_with_tools.ainvoke(messages)

    answer = response.content.strip()
    logs.append(_ts("general_node", f"Response ready ({len(answer)} chars)."))
    return {"final_answer": answer, "route_used": "general", "logs": logs}



# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — RAG & Research Node
# ══════════════════════════════════════════════════════════════════════════════

async def rag_node(state: UniversalAgentState) -> dict:
    """
    Retrieval-Augmented Generation:
    1. Search Qdrant for relevant chunks (from uploaded docs / past ingestion)
    2. If query mentions ArXiv / research papers → also search ArXiv
    3. Synthesise grounded answer with inline citations
    """
    query        = state["query"]
    uploaded     = state.get("uploaded_files", [])
    logs         = [_ts("rag_node", f"RAG retrieval for: '{query[:50]}'")]
    all_context  : list[str] = []
    all_sources  : list[dict] = []
    _, llm_smart = _get_llms(state.get("user_groq_key", ""))

    # ── Qdrant semantic search ────────────────────────────────────────────────
    filter_by = None
    if uploaded:
        # Filter to only the user's uploaded file IDs if provided
        file_ids = [f["file_id"] for f in uploaded if "file_id" in f]
        if file_ids:
            filter_by = {"file_id": file_ids[0]}  # primary file

    qdrant_hits = similarity_search(query, top_k=5, filter_payload=filter_by)
    for hit in qdrant_hits:
        all_context.append(hit["content"])
        all_sources.append({"type": "document", "score": hit["score"],
                             "file": hit.get("filename", "uploaded doc")})
    logs.append(_ts("rag_node", f"Qdrant → {len(qdrant_hits)} chunks retrieved."))

    # ── ArXiv search for research queries ─────────────────────────────────────
    research_keywords = ["paper", "research", "arxiv", "study", "survey", "model", "algorithm"]
    if any(kw in query.lower() for kw in research_keywords):
        papers = await search_arxiv(query, max_results=3)
        for p in papers:
            all_context.append(f"[ArXiv] {p['title']}: {p['abstract']}")
            all_sources.append({"type": "arxiv", "title": p["title"],
                                "url": p["pdf_url"], "year": p["year"]})
        logs.append(_ts("rag_node", f"ArXiv → {len(papers)} papers added."))

    # ── LLM synthesis ─────────────────────────────────────────────────────────
    context_block = "\n\n---\n\n".join(all_context[:8]) if all_context else "No relevant context found."

    system = (
        "You are a precise research assistant. Answer the user's question using ONLY "
        "the provided context. Cite sources as [Doc 1], [Doc 2] etc. "
        "If the context doesn't contain the answer, say so clearly. "
        "Use Markdown formatting with headers and bullet points where appropriate. "
        "Be thorough and detailed in your answer."
    )
    user_prompt = f"CONTEXT:\n{context_block[:6000]}\n\nQUESTION: {query}"

    response = await llm_smart.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])

    answer = response.content.strip()
    logs.append(_ts("rag_node", f"RAG answer generated ({len(answer)} chars)."))
    return {
        "final_answer": answer,
        "rag_context":  all_context,
        "sources":      all_sources,
        "route_used":   "rag",
        "logs":         logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Web & Search Node
# ══════════════════════════════════════════════════════════════════════════════

async def web_node(state: UniversalAgentState) -> dict:
    """
    Performs DuckDuckGo search and synthesises a grounded answer.
    Handles: news queries, weather, general web research.
    """
    query   = state["query"]
    logs    = [_ts("web_node", f"Web search: '{query[:50]}'")]
    previous_query = _last_user_message(state.get("messages", []))
    llm_fast, llm_smart = _get_llms(state.get("user_groq_key", ""))

    # ── Weather check: use structured Open-Meteo data ─────────────────────────
    is_weather = is_weather_query(query) or bool(
        previous_query
        and is_weather_query(previous_query)
        and len(query.split()) <= 10
    )

    if is_weather:
        logs.append(_ts("web_node", "Detected weather query — fetching live data..."))
        location = _extract_weather_location(query)
        if not location:
            loc_resp = await llm_fast.ainvoke([HumanMessage(
                content=(
                    "Extract ONLY the city/location name from this weather query. "
                    f"Reply with ONLY the location name.\nQuery: {query}"
                )
            )])
            location = loc_resp.content.strip()
        logs.append(_ts("web_node", f"Fetching weather for: {location}"))
        weather_data = await get_weather(location)
        answer = format_weather_result(weather_data)
        weather_source = weather_data.get("source") or {}
        sources = []
        if weather_source.get("weather_url"):
            sources.append({
                "type": "web",
                "title": "Open-Meteo live weather",
                "url": weather_source["weather_url"],
            })
        logs.append(_ts("web_node", f"Weather response generated."))
        return {
            "final_answer": answer,
            "sources":      sources,
            "route_used":   "web",
            "logs":         logs,
        }

    # ── Instagram check ───────────────────────────────────────────────────────
    if is_instagram_query(query):
        logs.append(_ts("web_node", "Detected Instagram query — fetching news..."))
        # Extract topic: strip instagram/insta from query to get the actual subject
        topic = re.sub(
            r"\b(?:instagram|insta|ig)\b",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip(" ,?.-")
        topic = re.sub(
            r"\b(?:news|trending|viral|posts?|reels?|stories|latest|update|what(?:'s)? (?:happening|new))\b",
            "",
            topic,
            flags=re.IGNORECASE,
        ).strip(" ,?.-")
        current_date = datetime.now().astimezone().date().isoformat()
        ig_results = await get_instagram_news(topic=topic, max_results=8)
        answer = format_instagram_results(ig_results, topic, current_date)
        sources = [
            {"type": "web", "title": r["title"], "url": r["url"], "published_at": r.get("published_at", "")}
            for r in ig_results if r.get("url")
        ]
        logs.append(_ts("web_node", f"Instagram: {len(ig_results)} results."))
        return {
            "final_answer": answer,
            "sources":      sources,
            "route_used":   "instagram",
            "logs":         logs,
        }

    # ── News check ────────────────────────────────────────────────────────────
    is_news = is_news_query(query) or bool(
        previous_query
        and is_news_query(previous_query)
        and len(query.split()) <= 10
    )
    current_date = datetime.now().astimezone().date().isoformat()

    if is_news:
        # Build a contextual query for follow-ups
        contextual_query = (
            f"{previous_query}. Follow-up: {query}"
            if previous_query and not is_news_query(query)
            else query
        )
        logs.append(_ts("web_node", "News query — using tiered news search (NewsAPI → Tavily → DDG)"))
        results = await news_search(contextual_query, max_results=8, freshness="d")
        answer = format_news_results(results, current_date)
        sources = [
            {
                "type": "web",
                "title": r["title"],
                "url": r["url"],
                "published_at": r.get("published_at", ""),
                "image_url": r.get("image_url", ""),
                "snippet": r.get("snippet", ""),
                "source": r.get("source", ""),
                "category": r.get("category", "news"),
            }
            for r in results if r.get("url")
        ]
        logs.append(_ts("web_node", f"News: {len(results)} articles retrieved."))
        return {
            "final_answer": answer,
            "sources": sources,
            "route_used": "web",
            "logs": logs,
        }

    # ── General web search (DuckDuckGo) ──────────────────────────────────────
    results = await web_search(query, max_results=8)
    formatted = format_search_results(results)
    sources = [
        {
            "type": "web",
            "title": r["title"],
            "url": r["url"],
            "published_at": r.get("published_at", ""),
            "image_url": "",
        }
        for r in results
    ]
    logs.append(_ts("web_node", f"Found {len(results)} web results."))

    if not results:
        return {
            "final_answer": (
                "I could not retrieve live web results right now. "
                "Please try again in a moment."
            ),
            "sources": [],
            "route_used": "web",
            "logs": logs,
        }

    system = (
        "You are a web research assistant. Synthesise the web search results below "
        "into a clear, accurate, well-structured answer. "
        f"The current local date is {current_date}. Prefer the newest reliable result. "
        "Include relevant URLs as markdown links. Do not hallucinate facts. "
        "Use headers, bullet points and markdown formatting where appropriate."
    )

    user_prompt = f"SEARCH RESULTS:\n{formatted}\n\nQUESTION: {query}"

    response = await llm_smart.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])

    answer = response.content.strip()
    logs.append(_ts("web_node", f"Web answer generated ({len(answer)} chars)."))
    return {
        "final_answer": answer,
        "sources":      sources,
        "route_used":   "web",
        "logs":         logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Finance & Live API Node
# ══════════════════════════════════════════════════════════════════════════════

async def finance_node(state: UniversalAgentState) -> dict:
    """
    Handles stock prices and live sports scores.
    Ticker is pre-extracted by supervisor_node if available.
    """
    query   = state["query"]
    logs    = [_ts("finance_node", f"Finance query: '{query[:50]}'")]
    sources = state.get("sources", [])
    answer  = ""
    llm_fast, _ = _get_llms(state.get("user_groq_key", ""))

    # Determine if this is stock or sports
    # Detect sports/cricket — use the same helper as the router for consistency
    _ql = query.lower()
    is_sports = is_cricket_score_query(query) or any(w in _ql for w in [
        "cricket", "ipl", "scorecard", "match", "football", "soccer",
        "test match", "odi", "t20", "live game", "live score", "live match",
    ])

    if is_sports:
        logs.append(_ts("finance_node", "Fetching live cricket scores..."))
        data   = await get_cricket_scores(query=query)
        answer = format_cricket_result(data)
        sources.append({"type": "sports", "source": "cricapi.com"})
    else:
        # Extract ticker from pre-computed sources or re-extract
        ticker = None
        for s in sources:
            if s.get("type") == "ticker":
                ticker = s["value"]
                break

        if not ticker:
            # Fallback extraction
            resp = await llm_fast.ainvoke([HumanMessage(
                content=f"What is the stock ticker symbol in this query? Reply ONLY the ticker.\nQuery: {query}"
            )])
            ticker = resp.content.strip().upper()

        logs.append(_ts("finance_node", f"Fetching stock data for: {ticker}"))
        stock_data = await get_stock_price(ticker)
        answer     = format_stock_result(stock_data)
        sources.append({"type": "stock", "ticker": ticker, "source": "yfinance"})

    # Enrich with LLM commentary
    enrichment = await llm_fast.ainvoke([
        SystemMessage(content="Add a brief 1-2 sentence financial insight or context to this data. Be concise."),
        HumanMessage(content=f"Data:\n{answer}\n\nOriginal question: {query}"),
    ])
    answer += f"\n\n> **💡 Insight**: {enrichment.content.strip()}"

    logs.append(_ts("finance_node", "Finance response complete."))
    return {
        "final_answer": answer,
        "sources":      sources,
        "route_used":   "finance",
        "logs":         logs,
    }



# ##1. NewsAPI (primary)   → 200 req/day, returns images + text
# 2. Tavily (fallback)   → kicks in when NewsAPI quota is hit
# 3. DuckDuckGo (backup) → last resort if Tavily also fails
# 4. Google News RSS     → final safety net

# ══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Farmer Node (Specialized Agriculture Assistant)
# ══════════════════════════════════════════════════════════════════════════════

async def farmer_node(state: UniversalAgentState) -> dict:
    """
    Dedicated node for Farmer Mode.
    Handles weather-aware farming advice, crop market prices, and govt schemes.
    """
    query = state["query"]
    logs = state.get("logs", [])
    sources = state.get("sources", [])
    logs.append(_ts("farmer_node", f"Farmer query: '{query[:50]}'"))
    llm_fast, llm_smart = _get_llms(state.get("user_groq_key", ""))

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %A")
    
    # Check if it's a weather query
    if is_weather_query(query):
        location = _extract_weather_location(query)
        if not location:
            location = "your region"
            
        logs.append(_ts("farmer_node", f"Fetching weather for: {location}"))
        weather_data = await get_weather(location)
        formatted_weather = format_weather_result(weather_data)
        
        system = (
            "You are a specialized Agricultural Assistant for farmers in India. "
            "You are provided with live weather data for the farmer's region. "
            "Provide actionable, weather-aware farming advice based on this forecast. "
            "For example: 'Delay pesticide spraying due to upcoming rain' or 'Good time to harvest'. "
            "Be practical, encouraging, and use simple language. "
            f"The current local date is {current_date}."
        )
        user_prompt = f"WEATHER DATA:\n{formatted_weather}\n\nFARMER QUESTION: {query}"
        sources.append({"type": "weather", "location": location})
        
    # Check if it's about crop prices/market
    elif re.search(r"\b(?:price|prices|mandi|rate|rates|market|buy|sell)\b", query, re.IGNORECASE):
        logs.append(_ts("farmer_node", "Fetching AGMARKNET mandi prices"))
        
        # Extract commodity and state using fast LLM
        resp = await llm_fast.ainvoke([HumanMessage(
            content=f"Extract the primary crop/commodity name AND the Indian state from this query. Correct any spelling (e.g. 'potatos' -> 'Potato'). Expand state abbreviations (e.g. 'UP' -> 'Uttar Pradesh'). Reply in format 'Commodity|State'. If state is unknown, reply 'Commodity|'. If commodity is unknown, reply 'Unknown|'.\nQuery: {query}"
        )])
        parts = resp.content.strip().split('|')
        commodity = parts[0].strip().title() if len(parts) > 0 else "Unknown"
        state_filter = parts[1].strip().title() if len(parts) > 1 else ""
        
        if commodity and commodity != "Unknown":
            mandi_data = await get_mandi_prices(commodity, state=state_filter, max_results=10)
            
            # If AGMARKNET fails or returns nothing, fallback to Web Search
            if "error" in mandi_data or not mandi_data.get("records"):
                logs.append(_ts("farmer_node", "AGMARKNET empty/failed, falling back to Web Search"))
                # Build a targeted search query instead of the raw user sentence
                enhanced_query = f"{commodity} price in {state_filter if state_filter else 'India'} mandi today"
                raw_results = await web_search(enhanced_query, max_results=5)
                formatted_data = format_search_results(raw_results)
                sources.extend(raw_results)
                data_context = f"AGMARKNET had no live data for this specific region today.\nLIVE WEB SEARCH RESULTS:\n{formatted_data}"
            else:
                formatted_data = format_mandi_prices(mandi_data, commodity)
                sources.append({"type": "mandi", "commodity": commodity, "source": "AGMARKNET"})
                data_context = f"AGMARKNET MANDI PRICES:\n{formatted_data}"
        else:
            # Fallback to web search if no commodity detected
            logs.append(_ts("farmer_node", "No commodity detected, falling back to Web Search"))
            clean_query = re.sub(r"^(please\s+|tell me\s+|tell\s+|what is\s+|find\s+|search for\s+|about\s+|can you\s+)+", "", query, flags=re.IGNORECASE).strip()
            enhanced_query = f"{clean_query} agriculture India latest"
            raw_results = await web_search(enhanced_query, max_results=5)
            formatted_data = format_search_results(raw_results)
            sources.extend(raw_results)
            data_context = f"LIVE WEB SEARCH RESULTS:\n{formatted_data}"
            
        system = (
            "You are a specialized Agricultural Assistant for farmers in India. "
            "You have live data regarding crop prices. "
            "Summarize the data clearly for the farmer. "
            "Highlight the crop name, location/mandi, and current rate. "
            "Use emojis (🌾💰🚜) and clear bullet points. Do not hallucinate data. "
            f"The current local date is {current_date}."
        )
        user_prompt = f"{data_context}\n\nFARMER QUESTION: {query}"
        
    # Check if it's about govt schemes, news, or updates
    elif re.search(r"\b(?:scheme|schemes|schems|government|govt|yojana|pm kisan|news|update|updates|subsidy|loan)\b", query, re.IGNORECASE):
        logs.append(_ts("farmer_node", "Fetching live scheme/news data via Web Search"))
        clean_query = re.sub(r"^(please\s+|tell me\s+|tell\s+|what is\s+|find\s+|search for\s+|about\s+|can you\s+)+", "", query, flags=re.IGNORECASE).strip()
        enhanced_query = f"{clean_query} agriculture India latest"
        raw_results = await web_search(enhanced_query, max_results=5)
        formatted_web = format_search_results(raw_results)
        
        system = (
            "You are a specialized Agricultural Assistant for farmers in India. "
            "You have performed a web search to find government scheme details or agricultural news. "
            "Summarize the search results clearly for the farmer. "
            "Explain the benefits simply and how to apply (if a scheme). "
            "Use emojis (🌾💰🚜) and clear bullet points. Do not hallucinate data. "
            f"The current local date is {current_date}."
        )
        user_prompt = f"LIVE WEB SEARCH RESULTS:\n{formatted_web}\n\nFARMER QUESTION: {query}"
        sources.extend(raw_results)
        
    # General farming/basic knowledge fallback
    else:
        logs.append(_ts("farmer_node", "Handling general farming question"))
        system = (
            "You are a specialized Agricultural Assistant for farmers in India. "
            "Answer the farmer's question using your basic knowledge. "
            "Focus on crop health, soil management, fertilizers, and best practices. "
            "Be respectful, practical, and use simple language. "
        )
        user_prompt = f"FARMER QUESTION: {query}"

    # Build messages list including conversation history
    history = state.get("messages", [])
    
    # Convert dict history to LangChain message objects
    langchain_history = []
    for msg in history:
        if msg.get("role") == "user":
            langchain_history.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            from langchain_core.messages import AIMessage
            langchain_history.append(AIMessage(content=msg.get("content", "")))

    messages_to_send = [SystemMessage(content=system)] + langchain_history + [HumanMessage(content=user_prompt)]

    response = await llm_smart.ainvoke(messages_to_send)

    answer = response.content.strip()
    logs.append(_ts("farmer_node", "Farmer response complete."))
    
    return {
        "final_answer": answer,
        "sources": sources,
        "route_used": "farmer",
        "logs": logs,
    }
