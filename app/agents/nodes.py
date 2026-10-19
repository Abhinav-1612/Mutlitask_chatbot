"""
app/agents/nodes.py — LangGraph Agent Nodes (Ultra-Low Latency Architecture)
=============================================================================
1. agent_node: Omni-Agent that binds real-time tools (weather, news, stocks, cricket, arxiv, PDF search)
2. farmer_node: Dedicated agricultural specialist with weather advice, Mandi prices, and govt schemes
"""
from __future__ import annotations

import logging
import json
import re
import time
import asyncio
from datetime import datetime, timezone
from typing import Literal, Any

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
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
_rate_limited_models: dict[str, float] = {}  # model_name -> expiration_timestamp

def _get_active_model(requested_model: str) -> str:
    now_ts = time.time()
    req = requested_model.strip() or settings.agent_model
    if req in _rate_limited_models and now_ts < _rate_limited_models[req]:
        # Return fallback model while requested model is cooling down
        return "qwen/qwen3.6-27b" if "gpt-oss" in req else "openai/gpt-oss-120b"
    return req

def _get_llms(user_key: str = "", agent_model: str = "") -> tuple:
    """
    Return (llm_fast, llm_smart) for the given API key and model.
    Instances are cached per key+model so we don’t recreate on every token.
    Falls back to the server’s GROQ_API_KEY if user_key is empty.
    """
    api_key = (user_key.strip() or settings.groq_api_key)
    raw_model = (agent_model.strip() or settings.agent_model)
    model_name = _get_active_model(raw_model)
    cache_key = f"{api_key}_{model_name}"

    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = (
            ChatGroq(
                model=settings.router_model,
                api_key=api_key,
                temperature=0.0,
                max_tokens=512,
                max_retries=1,
            ),
            ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0.3,
                max_tokens=2048,
                max_retries=1,
            ),
        )
    return _llm_cache[cache_key]

# ── Tools for Omni-Agent ──────────────────────────────────────────────────────

class WeatherInput(BaseModel):
    location: str = Field(description="The exact name of the city, state, or country ONLY. Example: 'Nagpur' or 'Tokyo'.")

@tool(args_schema=WeatherInput)
async def tool_get_weather(location: str) -> str:
    """Get the current weather and 3-day forecast for any city or location."""
    try:
        data = await get_weather(location)
        return format_weather_result(data)
    except Exception as e:
        return f"Weather Error: {str(e)}"

class NewsInput(BaseModel):
    topic_or_location: str = Field(description="The topic, city/location, or company to fetch today's news and top headlines for. Example: 'Mumbai', 'Microsoft', or 'Tesla'.")

@tool(args_schema=NewsInput)
async def tool_search_news(topic_or_location: str) -> str:
    """Fetch today's breaking news, top headlines, and current news articles for any city, country, company, or topic."""
    try:
        results = await news_search(topic_or_location, max_results=7, freshness="d")
        return format_news_results(results, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    except Exception as e:
        return f"News Error: {str(e)}"

class WebSearchInput(BaseModel):
    query: str = Field(
        description=(
            "The complete, self-contained web search query string. "
            "ALWAYS include the full context — never use short phrases like 'the fee' or 'more info'. "
            "Example: 'GATE 2027 application fee amount general category', "
            "'Vivek Yadav student IIIT Nagpur profile LinkedIn', "
            "'director of IIT Bombay 2026'. "
            "This field is REQUIRED and must always be a non-empty string."
        )
    )

@tool(args_schema=WebSearchInput)
async def tool_web_search(query: str) -> str:
    """Search the web for any facts, questions, people, student profiles, institutional leadership, movies, reviews, and events. ALWAYS pass a full descriptive query string."""
    try:
        if is_news_query(query):
            results = await news_search(query, max_results=7, freshness="d")
            return format_news_results(results, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        else:
            results = await web_search(query, max_results=5)
            return format_search_results(results) if results else "No results found."
    except Exception as e:
        return f"Search Error: {str(e)}"

class WikipediaInput(BaseModel):
    query: str = Field(description="The topic to search for on Wikipedia. Example: 'Toxic (2026 film)', 'Quantum computing', etc.")

@tool(args_schema=WikipediaInput)
async def tool_search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual summaries, biographies, movies, or historical events."""
    try:
        from app.tools.search import search_wikipedia
        results = await search_wikipedia(query, max_results=3)
        if not results:
            return "No Wikipedia articles found."
        lines = []
        for index, res in enumerate(results, 1):
            lines.append(f"**[{index}] {res.get('title')}**")
            lines.append(f"URL: {res.get('url')}")
            lines.append(f"{res.get('snippet')}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Wikipedia Error: {str(e)}"

class StockInput(BaseModel):
    ticker_or_company: str = Field(description="The stock ticker symbol or company name. Example: 'TSLA', 'Tesla', 'AAPL', or 'RELIANCE.NS'.")

@tool(args_schema=StockInput)
async def tool_get_stock_price(ticker_or_company: str) -> str:
    """Fetch live and historical stock data (current price, yesterday's exact closing price / previous close, change %, 52W range) for any company or ticker symbol."""
    try:
        clean_ticker = _extract_clean_ticker(ticker_or_company, ticker_or_company) or ticker_or_company
        data = await get_stock_price(clean_ticker)
        return format_stock_result(data)
    except Exception as e:
        return f"Finance Error: {str(e)}"

class CricketInput(BaseModel):
    query: str = Field(description="The cricket match query (e.g., 'India live score', 'India vs Sri Lanka match result', 'IPL today', 'Ind vs SL summary').")

@tool(args_schema=CricketInput)
async def tool_get_cricket_scores(query: str) -> str:
    """Fetch live cricket scores, match updates, recent match results, summaries, and scorecards."""
    try:
        data = await get_cricket_scores(query)
        return format_cricket_result(data)
    except Exception as e:
        return f"Sports Error: {str(e)}"

class ArxivInput(BaseModel):
    query: str = Field(description="The research topic or paper title to search on arXiv.")

@tool(args_schema=ArxivInput)
async def tool_search_arxiv(query: str) -> str:
    """Search for academic research papers on arXiv."""
    try:
        papers = await search_arxiv(query, max_results=3)
        return format_arxiv_results(papers) if papers else "No papers found."
    except Exception as e:
        return f"ArXiv Error: {str(e)}"

# ── Helper ─────────────────────────────────────────────────────────────────────
def _ts(node: str, msg: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}][{node}] {msg}"
    logger.info(entry)
    return entry


async def _emit_step(state: UniversalAgentState, node: str, message: str, logs_list: list[str] | None = None) -> None:
    entry = _ts(node, message)
    if logs_list is not None:
        logs_list.append(entry)
    queue = state.get("event_queue")
    if queue:
        try:
            await queue.put({"node": node, "message": message, "raw": entry})
        except Exception:
            pass


def _clean_llm_response(text: str) -> str:
    """Remove internal reasoning or thinking process artifacts from LLM response."""
    if not text:
        return ""
    
    # 1. Remove <think>...</think> tags (case-insensitive)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

    # 2. Check for Chain-of-Thought (CoT) planning monologue
    cot_patterns = [
        r"Analyze (?:the )?(?:Request|User Input|Retrieved Data):",
        r"Drafting (?:the )?Response",
        r"Review against Constraints",
        r"Final Polish",
        r"Task \d+:",
        r"Constraint:",
        r"Thinking Process:"
    ]
    has_cot = any(re.search(p, text, re.IGNORECASE) for p in cot_patterns)

    if has_cot:
        # Replace "Visual Header: ###" or "Header: ###" with "###"
        text = re.sub(r"(?:Visual\s+)?Header:\s*(###\s+)", r"\1", text, flags=re.IGNORECASE)
        
        # If there are markdown headers `### `
        headers = list(re.finditer(r"(?m)^###\s+", text))
        if headers:
            best_start = headers[0].start()
            for h in headers:
                after = text[h.start():h.start()+200]
                if not any(re.search(p, after, re.IGNORECASE) for p in [r"Table \d+:", r"Structured Tables:", r"Constraint:"]):
                    best_start = h.start()
                    break
            text = text[best_start:]
        elif "--- RETRIEVED" not in text and "|" in text:
            m = re.search(r"(?m)^(\|[^\n]+\|\n\|[\s\-:]+\|\n|\*\*[^\n]+\*\*|#[^\n]+)", text)
            if m:
                text = text[m.start():]

    # 3. Clean up any leftover planning artifacts within the text
    text = re.sub(r"(?m)^(?:Visual Header|Header|Structured Tables|Narrative Highlights|News & Story Cards|Interactive Links|Strict Factual Grounding|Direct Response Only|Complete Coverage|Review against Constraints):\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^Poem:\s*(?:Need a short poem[^\n]*\n)?", "### 🐾 Poem\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^Table:\s*", "", text, flags=re.IGNORECASE)

    # 4. Strip reasoning / self-correction / CoT markers
    split_markers = [
        r"\[Output Generation\]\s*(?:\(Proceeds\))?\s*(?:\(Note:[^\)]*\))?\s*",
        r"\[Final Answer\]:?\s*",
        r"\[Output\]:?\s*",
        r"(?:^|\n)(?:Here['’]?s (?:a )?thinking process|Thinking Process):.*?(?=\n\n|\Z)",
        r"(?:^|\n)(?:Self-Correction[^\n]*):.*?(?=\n\n|\Z)",
        r"Output matches\.?✅?\s*Done\.?\s*",
    ]
    for marker in split_markers:
        parts = re.split(marker, text, flags=re.IGNORECASE | re.DOTALL)
        if len(parts) > 1:
            for part in reversed(parts):
                p_clean = part.strip()
                if len(p_clean) > 30 and not p_clean.startswith("[") and not p_clean.startswith("("):
                    text = p_clean
                    break

    # 5. Remove search/citation artifacts like 【1†L1-L3】
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[Done\].*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\(Self-Correction.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\(Note: I will output.*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"【.*?】", "", text)
    
    # 6. Fix corrupted or nested markdown link syntax
    text = re.sub(r'\[+([^\]]+)\]+\(\[[^\]]+\]\((https?://[^\s\)]+)\)\)', r'[\1](\2)', text)
    text = re.sub(r'\[+([^\]]+)\]+\(\[(https?://[^\s\]]+)\]\((https?://[^\s\)]+)\)\)', r'[\1](\2)', text)
    text = re.sub(r'\[+([^\]]+)\]+\(\[(https?://[^\s\]]+)\]\)', r'[\1](\2)', text)
    text = re.sub(r'\[\\?\[([^\]]+)\\?\]\]\((https?://[^\s\)]+)\)', r'[\1](\2)', text)
    text = re.sub(r'\[\[(https?://[^\s\]]+)\]\]\((https?://[^\s\)]+)\)', r'[Read article →](\1)', text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"[\1]", text)

    return text.strip()


# Common company to ticker mapping for instantaneous, error-free resolution
COMMON_TICKER_MAP = {
    "TESLA": "TSLA",
    "TSLA": "TSLA",
    "APPLE": "AAPL",
    "AAPL": "AAPL",
    "MICROSOFT": "MSFT",
    "MSFT": "MSFT",
    "GOOGLE": "GOOGL",
    "GOOGL": "GOOGL",
    "GOOG": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "AMZN": "AMZN",
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "NFLX": "NFLX",
    "AMD": "AMD",
    "INTEL": "INTC",
    "INTC": "INTC",
    "TSMC": "TSM",
    "TSM": "TSM",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "WIPRO": "WIPRO.NS",
    "ITC": "ITC.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "ADANI": "ADANIENT.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "SWIGGY": "SWIGGY.NS",
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHEREUM": "ETH-USD",
}


def _extract_clean_ticker(raw_text: str, query: str = "") -> str:
    """Robustly extract and clean ticker symbol from LLM output or user query."""
    if query:
        q_upper = query.upper()
        for comp in sorted(COMMON_TICKER_MAP.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(comp)}\b", q_upper):
                return COMMON_TICKER_MAP[comp]

    if not raw_text:
        return ""

    cleaned = _clean_llm_response(raw_text)
    c_upper = cleaned.upper()
    for comp in sorted(COMMON_TICKER_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(comp)}\b", c_upper):
            return COMMON_TICKER_MAP[comp]

    cleaned = re.sub(r"```[\w]*", "", cleaned)
    cleaned = re.sub(r"[`\"'*_#:]", " ", cleaned)

    tokens = re.findall(r"[A-Za-z0-9.\-^]{1,12}", cleaned)
    ignore_words = {
        "THE", "TICKER", "SYMBOL", "FOR", "IS", "STOCK", "OF", "COMPANY", "REPLY",
        "ONLY", "NONE", "PRICE", "QUOTE", "EXCHANGE", "NASDAQ", "NYSE", "BSE",
        "NSE", "HERE", "WHAT", "CURRENT", "LIVE", "SHARE", "VALUE", "OUTPUT", "FINAL", "YES", "NO"
    }

    for token in tokens:
        candidate = token.strip().upper()
        if candidate and candidate not in ignore_words and not candidate.isdigit():
            sanitized = re.sub(r"[^A-Z0-9.\-^]", "", candidate)
            if sanitized and len(sanitized) <= 12 and sanitized != "NONE":
                return sanitized

    if query:
        q_cleaned = re.sub(r"[`\"'*_#:]", " ", query)
        q_tokens = re.findall(r"\b[A-Za-z0-9.\-^]{1,12}\b", q_cleaned)
        for token in q_tokens:
            candidate = token.strip().upper()
            if candidate and candidate not in ignore_words and not candidate.isdigit() and len(candidate) >= 2:
                sanitized = re.sub(r"[^A-Z0-9.\-^]", "", candidate)
                if sanitized and sanitized != "NONE":
                    return sanitized

    return ""


def _extract_weather_location(query: str) -> str:
    """Extract location name from a weather query robustly."""
    followup = re.search(
        r"^\s*(?:what|how)\s+about\s+(.+?)[?.!]*$",
        query,
        flags=re.IGNORECASE,
    )
    if followup:
        return followup.group(1).strip(" ,?.!")

    match = re.search(
        r"\b(?:of|in|for|at)\s+([A-Za-z][\w\s,]+?)(?:\s+(?:today|tomorrow|tonight|right\s*now|now|at\s*current|current|currently))?[?.!]*$",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" ,?.!")

    cleaned = re.sub(
        r"\b(?:what(?:'s| is)?|how(?:'s| is)?|the|current|currently|at current|weather|temperature|forecast|"
        r"humidity|conditions?|today|tomorrow|tonight|right\s*now|now|please|tell|me|of|is|get|india)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,?.!")


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY HELPER — Hybrid Low-Latency Long-Term Memory
# ══════════════════════════════════════════════════════════════════════════════

def _build_hybrid_history(
    messages: list[dict],
    uploaded_files: list[dict] | None = None,
) -> tuple[str, list[HumanMessage | AIMessage]]:
    """
    Builds an ultra-low latency, high-precision hybrid memory representation:
    1. A concise, structured long-term summary of older turns (> 6 turns ago) + uploaded document facts.
    2. The last 6 full verbatim messages as LangChain message objects.
    
    Zero latency overhead (0.00s deterministic extraction), preserving long-term memory for both
    conversations and uploaded PDFs without token bloat.
    """
    uploaded_files = uploaded_files or []
    summary_blocks = []

    # 1. Permanent Document Context Memory
    if uploaded_files:
        docs = [f.get("filename") or f.get("file_id", "document") for f in uploaded_files]
        summary_blocks.append(f"• Active Uploaded Document(s): {', '.join(docs)} (available for semantic search)")

    if not messages:
        long_term_summary = ""
        if summary_blocks:
            long_term_summary = (
                "\n[LONG-TERM CONVERSATION & DOCUMENT MEMORY]:\n"
                + "\n".join(summary_blocks)
                + "\n"
            )
        return long_term_summary, []

    # Split into older turns and recent 6 turns
    if len(messages) > 6:
        older_messages = messages[:-6]
        recent_messages = messages[-6:]
    else:
        older_messages = []
        recent_messages = messages

    # 2. Older Conversation Summary (queries, answers, user details, PDF findings)
    if older_messages:
        qa_summaries = []
        for i in range(len(older_messages)):
            msg = older_messages[i]
            if msg.get("role") == "user":
                u_text = msg.get("content", "").strip().replace("\n", " ")
                if len(u_text) > 110:
                    u_text = u_text[:110] + "..."
                a_snippet = ""
                if i + 1 < len(older_messages) and older_messages[i + 1].get("role") == "assistant":
                    a_raw = older_messages[i + 1].get("content", "").strip()
                    # Strip any HTML tags or markdown image blocks
                    a_clean = re.sub(r"<[^>]+>", "", a_raw)
                    a_clean = re.sub(r"!\[.*?\]\(.*?\)", "", a_clean).strip().replace("\n", " ")
                    if len(a_clean) > 150:
                        a_clean = a_clean[:150] + "..."
                    a_snippet = f" -> Key Result: {a_clean}"
                qa_summaries.append(f"  - User Query: \"{u_text}\"{a_snippet}")

        if qa_summaries:
            summary_blocks.append(
                "• Prior Conversation History & Established Facts:\n"
                + "\n".join(qa_summaries[-10:])
            )

    long_term_summary = ""
    if summary_blocks:
        long_term_summary = (
            "\n[LONG-TERM CONVERSATION & DOCUMENT MEMORY]:\n"
            + "\n".join(summary_blocks)
            + "\nUse this memory context to maintain continuity with user facts, preferences, and previously discussed topics.\n"
        )

    # 3. Convert recent messages into LangChain messages
    langchain_recent = []
    for msg in recent_messages:
        content = msg.get("content", "")
        if msg.get("role") == "assistant" and len(content) > 1200:
            content = content[:1200] + "\n...(truncated for context)"
        if msg.get("role") == "user":
            langchain_recent.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            langchain_recent.append(AIMessage(content=content))

    return long_term_summary, langchain_recent


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Omni Agent Node (Ultra-Low Latency ReAct Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

async def agent_node(state: UniversalAgentState) -> dict:
    """
    Omni-Agent that natively binds all tools and executes them in parallel if needed.
    Optimized for single-pass parallel tool calling and pristine synthesis with gpt-oss-120b.
    """
    query = state["query"]
    logs: list[str] = []
    sources = state.get("sources", [])
    total_tokens = 0
    tools_called = state.get("tools_called", [])
    user_key = state.get("user_groq_key", "")
    agent_model = state.get("agent_model", "") or settings.agent_model

    # Clear any stale short-lived rate-limit bans (use 30s not 600s)
    now_ts = time.time()
    stale = [m for m, exp in _rate_limited_models.items() if now_ts >= exp]
    for m in stale:
        del _rate_limited_models[m]

    # Always use gpt-oss-120b unless it's currently cooling down
    active_model = agent_model
    if active_model in _rate_limited_models:
        # Use qwen only as a short temporary fallback
        active_model = "qwen/qwen3.6-27b" if "gpt-oss" in active_model else "openai/gpt-oss-120b"

    api_key = (user_key.strip() or settings.groq_api_key)

    # llm_tool: used for tool calling (bound with tools, parallel dispatch in system prompt)
    # llm_synth: plain (no tools bound), used exclusively for synthesis — never switches to Qwen
    # max_tokens=1024 ensures 3+ parallel tool calls (each ~200-300 tokens) are fully emitted
    llm_tool = ChatGroq(
        model=active_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=1024,
        max_retries=1,
    )
    llm_synth = ChatGroq(
        model=agent_model if agent_model not in _rate_limited_models else active_model,
        api_key=api_key,
        temperature=0.3,
        max_tokens=4096,
        max_retries=1,
    )

    await _emit_step(state, "Omni-Agent Node", "🧠 Analyzing query intent & selecting appropriate tools...", logs)

    uploaded = state.get("uploaded_files", [])
    file_ids = [f["file_id"] for f in uploaded if "file_id" in f]

    # Base tools available
    _tools = [
        tool_get_weather, tool_search_news, tool_web_search,
        tool_get_stock_price, tool_get_cricket_scores, tool_search_arxiv,
        tool_search_wikipedia
    ]

    # Only bind document tool if a file was actually uploaded
    if file_ids:
        @tool
        async def tool_search_document(search_query: str) -> str:
            """Search the currently uploaded PDF document for context, facts, personal information, etc."""
            try:
                hits = similarity_search(search_query, top_k=5, filter_payload={"file_id": file_ids[0]})
                if not hits:
                    return "No relevant information found in the document."
                return "\n\n---\n\n".join(hit["content"] for hit in hits)
            except Exception as e:
                return f"Document Search Error: {str(e)}"
        _tools.append(tool_search_document)

    llm_with_tools = llm_tool.bind_tools(_tools)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    doc_rule = "• `tool_search_document`: Query the user's uploaded document or PDF.\n" if file_ids else ""

    # Build hybrid memory
    long_term_summary, langchain_history = _build_hybrid_history(
        state.get("messages", []),
        uploaded
    )

    system_prompt = (
        "You are Omni-Agent, a top-tier fast AI assistant with real-time tool access.\n"
        f"Today's date and time (UTC): {now}.\n"
        f"{long_term_summary}\n"
        "AVAILABLE TOOLS:\n"
        "• `tool_get_weather`: Current weather, temperature, forecasts for any location.\n"
        "• `tool_search_news`: Breaking news, today's headlines, regional/city/global updates.\n"
        "• `tool_get_stock_price`: Live and historical stock/ticker quotes, 52W range, market changes.\n"
        "• `tool_get_cricket_scores`: Live cricket scores, recent match results, match summaries, commentary, and scorecards.\n"
        "• `tool_search_arxiv`: Academic papers and scientific research.\n"
        "• `tool_search_wikipedia`: Dedicated Wikipedia search for biographies, institutions, and facts.\n"
        "• `tool_web_search`: General web search for ANY query about real-world facts, people, students, professionals, LinkedIn profiles, GitHub profiles, biographies, institutes, leadership (directors, CEOs), movies, reviews, and events.\n"
        f"{doc_rule}\n"
        "CRITICAL TOOL INSTRUCTIONS:\n"
        "1. ALWAYS USE TOOLS FOR PEOPLE/PERSON QUERIES: If the user asks about any person — student, professional, employee, researcher, faculty, athlete, celebrity — ALWAYS call `tool_web_search` immediately. NEVER refuse or say you cannot help. Search for them and report what is found.\n"
        "2. MULTI-INTENT & PARALLEL CALLS (MANDATORY): If the user asks for multiple things (e.g., 'news + weather + stock price', 'arXiv papers + weather + BTC price', 'weather in Paris and tech headlines'), you MUST invoke ALL required tools SIMULTANEOUSLY in a SINGLE tool call batch. NEVER call just one tool when multiple are needed. Emit ALL tool calls together at once.\n"
        "3. GREETINGS & APPRECIATION (MANDATORY): If the user says hi, hello, thanks, thank you, okay thanks, ok great, awesome, etc. — ALWAYS respond warmly and conversationally like ChatGPT. Acknowledge their message, then ask if there's anything else you can help with. NEVER give a blank or empty response to a greeting or appreciation.\n"
        "4. DIRECT ANSWER: If the user asks general coding, math, explanations, creative writing, or conversational questions without needing real-time data, respond directly without tools.\n"
        "5. DETAILED RESPONSES: If the user asks for a detailed summary, full explanation, or comprehensive analysis — provide COMPLETE, thorough answers. Never truncate or give a brief summary when detail is requested."
    )

    langchain_messages = [SystemMessage(content=system_prompt)] + langchain_history + [HumanMessage(content=query)]

    route_used = "general"
    tool_contexts = []

    async def _dispatch_and_execute(messages: list) -> tuple[any, bool]:
        """Single LLM call + parallel tool execution. Returns (response, had_tools)."""
        nonlocal total_tokens, route_used
        try:
            resp = await llm_with_tools.ainvoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            logger.error("[agent_node] LLM invoke failed: %s", e)
            if "rate limit" in err_str or "429" in err_str or "quota" in err_str:
                _rate_limited_models[active_model] = time.time() + 30
                # ONLY fallback to Qwen if the current model is gpt-oss
                # If the user selected Qwen and Qwen hits rate limit, we don't force gpt-oss.
                alt_model = "qwen/qwen3.6-27b" if "gpt-oss" in active_model else "openai/gpt-oss-120b"
                
                # Update: only log warning and switch if the active model was gpt-oss. If user selected Qwen, keep trying Qwen or fail.
                # Actually, let's keep the fallback logic robust: if model A fails, try model B.
                await _emit_step(state, "Omni-Agent Node", f"⚠️ Rate limit hit on {active_model} — retrying with {alt_model}...", logs)
                
                llm_fb = ChatGroq(model=alt_model, api_key=api_key, temperature=0.0, max_tokens=1024, max_retries=1)
                try:
                    resp = await llm_fb.bind_tools(_tools).ainvoke(messages)
                except Exception as e2:
                    raise e2
            else:
                raise e

        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            total_tokens = resp.usage_metadata.get("output_tokens", 0) or resp.usage_metadata.get("total_tokens", 0)

        if not resp.tool_calls:
            return resp, False

        # Validate tool calls — drop any with obviously wrong/missing args
        valid_tool_calls = []
        for tc in resp.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            # Validate required args for known tools
            if name == "tool_web_search" and not args.get("query"):
                logger.warning("[agent_node] Dropped invalid tool call %s with args %s — missing 'query'", name, args)
                continue
            if name == "tool_search_news" and not args.get("topic_or_location"):
                logger.warning("[agent_node] Dropped invalid tool call %s — missing 'topic_or_location'", name)
                continue
            if name == "tool_get_weather" and not args.get("location"):
                logger.warning("[agent_node] Dropped invalid tool call %s — missing 'location'", name)
                continue
            valid_tool_calls.append(tc)

        if not valid_tool_calls:
            # All tool calls were invalid — treat as a direct text response
            logger.warning("[agent_node] All tool calls were invalid; treating as direct response")
            return resp, False

        # Build parallel tasks for ALL tool calls in this response
        messages.append(resp)
        tasks = []
        for tc in valid_tool_calls:
            name = tc["name"]
            tools_called.append(name)
            args = tc.get("args", {})
            args_str = ", ".join(f"{k}='{v}'" for k, v in args.items()) if args else ""
            await _emit_step(state, "Tools Execution Node", f"🔧 Calling: {name}({args_str})", logs)

            if name == "tool_search_news":
                async def run_news(tc_args=args):
                    topic = tc_args.get("topic_or_location", "")
                    res = await news_search(topic, max_results=7, freshness="d")
                    for r in res: r["type"] = "web"
                    sources.extend(res)
                    return format_news_results(res, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
                tasks.append((tc["id"], name, run_news()))
            elif name == "tool_web_search":
                async def run_web(tc_args=args):
                    q = tc_args.get("query", "")
                    if is_news_query(q):
                        res = await news_search(q, max_results=7, freshness="d")
                        for r in res: r["type"] = "web"
                        sources.extend(res)
                        return format_news_results(res, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
                    else:
                        res = await web_search(q, max_results=5)
                        for r in res: r["type"] = "web"
                        sources.extend(res)
                        return format_search_results(res) if res else "No results found."
                tasks.append((tc["id"], name, run_web()))
            elif name == "tool_search_arxiv":
                async def run_arxiv(tc_args=args):
                    q = tc_args.get("query", "")
                    res = await search_arxiv(q, max_results=3)
                    for r in res: r["type"] = "arxiv"
                    sources.extend(res)
                    return format_arxiv_results(res) if res else "No papers found."
                tasks.append((tc["id"], name, run_arxiv()))
            elif name == "tool_search_wikipedia":
                async def run_wikipedia(tc_args=args):
                    q = tc_args.get("query", "")
                    from app.tools.search import search_wikipedia
                    res = await search_wikipedia(q, max_results=3)
                    for r in res: r["type"] = "web"
                    sources.extend(res)
                    if not res: return "No Wikipedia articles found."
                    lines = []
                    for i, r in enumerate(res, 1):
                        lines.append(f"**[{i}] {r.get('title')}**")
                        lines.append(f"URL: {r.get('url')}")
                        lines.append(f"{r.get('snippet')}\n")
                    return "\n".join(lines)
                tasks.append((tc["id"], name, run_wikipedia()))
            else:
                tool_func = next((t for t in _tools if t.name == name), None)
                if tool_func:
                    tasks.append((tc["id"], name, tool_func.ainvoke(args)))
                else:
                    async def err_func(n=name): return f"Unknown tool: {n}"
                    tasks.append((tc["id"], name, err_func()))

        # Execute ALL tools in this batch concurrently
        results = await asyncio.gather(*(t[2] for t in tasks), return_exceptions=True)

        for tc_id, name, result in zip(
            (t[0] for t in tasks), (t[1] for t in tasks), results
        ):
            if isinstance(result, Exception):
                logger.error("[agent_node] Tool %s failed: %s", name, result)
                result = f"Tool execution failed: {str(result)}"
            content_str = str(result)
            if len(content_str) > 1500:
                content_str = content_str[:1500] + "\n...(truncated for brevity)"
            messages.append(ToolMessage(content=content_str, name=name, tool_call_id=tc_id))
            tool_contexts.append(f"### Tool: {name}\n{content_str}")
            if "document" in name or "arxiv" in name or "rag" in name or "pdf" in name:
                route_used = "rag"
            elif "stock" in name or "cricket" in name or "finance" in name:
                route_used = "finance"
            elif "weather" in name or "search" in name or "news" in name or "web" in name:
                route_used = "web"
        return resp, True

    # ── Multi-Turn Parallel ReAct Loop (max 3 iterations) ───────────────────
    # Each iteration: LLM call → parallel tool execution → next LLM call
    # gpt-oss-120b: typically finishes in 1 iteration (all tools at once)
    # Qwen: may take 2-3 iterations (sequential tool calls, parallel within each)
    max_iterations = 3
    for iteration in range(max_iterations):
        try:
            response, had_tools = await _dispatch_and_execute(langchain_messages)
        except Exception as e:
            await _emit_step(state, "Omni-Agent Node", f"❌ Error: {str(e)}", logs)
            return {
                "final_answer": f"⚠️ Error: {str(e)}. Please try again shortly.",
                "route_used": "general", "sources": [], "logs": logs,
                "total_tokens": 0, "tools_called": tools_called
            }
        if not had_tools:
            # Model returned a final text answer — done
            langchain_messages.append(response)
            break

    # ── Synthesis: runs once after the loop collects all tool results ────────
    if tool_contexts:
        await _emit_step(state, "Data Retrieval Node", f"📊 Retrieved data from {len(tool_contexts)} tool(s) successfully.", logs)
        await _emit_step(state, "Synthesis Node", "⚡ Synthesizing verified final response & citations...", logs)

        combined_tool_context = "\n\n---\n\n".join(tool_contexts)

        synthesis_system_prompt = (
            "You are Omni-Agent, a top-tier AI assistant synthesizing real-time data into a polished, high-end response.\n"
            f"Today's date and time (UTC): {now}.\n"
            "You are provided with real-time data retrieved from tools. Your job is to craft a comprehensive, beautifully structured final response.\n\n"
            "FORMATTING GUIDELINES:\n"
            "1. Visual Header: Start your response with a bold, topic-specific title with emoji (e.g. `### 📰 Today's News Headlines: Varanasi, UP`, `### 📈 MSFT Stock Snapshot`, `### 🎬 ...`).\n"
            "2. News Queries (STRICT STORY CARDS FORMAT): When answering news questions (e.g. city news, national news, topic news):\n"
            "   • DO NOT generate tables for news! Never create 'Topic | Highlights' or 'Category | Summary' tables for news.\n"
            "   • Include 2-3 brief bullet points under 'Key Highlights' summarizing today's key developments.\n"
            "   • Display ALL distinct news stories retrieved (aim for 5 to 7 articles). Format EACH article as:\n\n"
            "     **📰 Article Headline Here**\n"
            "     Concise 1-2 sentence summary of what happened.\n"
            "     [Read on SourceName →](URL)\n\n"
            "     ---\n\n"
            "3. Structured Tables (FOR METRICS & NON-NEWS DATA): Use Markdown Tables ONLY for stock prices, financial statistics, weather metrics, sports scores, and comparisons:\n\n"
            "   | Parameter / Metric | Details |\n"
            "   | :--- | :--- |\n"
            "   | **Key Attribute** | Value / Description |\n\n"
            "4. Interactive Links: Format all URLs cleanly as `[Read on {Source} →]({URL})`. CRITICAL: ONLY use URLs that appear VERBATIM in the retrieved tool data above. NEVER invent, guess, or construct URLs (especially GitHub, LinkedIn, or social media links) that are not present in the retrieved data. If no URL is available, omit the link entirely.\n"
            "5. Strict Factual Grounding: State facts strictly based on the retrieved context. If the tool data says a person was not found or results are limited, say so honestly — do NOT invent profile details.\n"
            "6. Complete Coverage: Address all aspects of the user prompt thoroughly.\n"
            "7. Direct Output: Output ONLY the final response in Markdown. No internal thinking, no preamble."
        )

        synthesis_messages = [
            SystemMessage(content=synthesis_system_prompt),
            *langchain_history,
            HumanMessage(
                content=(
                    f"User Query: {query}\n\n"
                    f"--- RETRIEVED REAL-TIME DATA ---\n"
                    f"{combined_tool_context}\n"
                    f"--------------------------------\n\n"
                    f"Please synthesize the final response based on the retrieved real-time data above."
                )
            ),
        ]

        try:
            # Use llm_synth directly (no tools bound — prevents tool hallucination)
            final_response = await llm_synth.ainvoke(synthesis_messages)
            if not getattr(final_response, "content", None) or not str(final_response.content).strip():
                retry_msgs = synthesis_messages + [
                    SystemMessage(content="CRITICAL: Output the answer directly in Markdown now. No internal thinking.")
                ]
                final_response = await llm_synth.ainvoke(retry_msgs)
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "quota" in err_str:
                # If active model hit a limit, swap to the alternative for synthesis
                synth_fallback_model = "qwen/qwen3.6-27b" if "gpt-oss" in active_model else "openai/gpt-oss-120b"
                await _emit_step(state, "Omni-Agent Node", f"⚠️ Synthesis rate limit hit on {active_model} — trying {synth_fallback_model}...", logs)
                llm_synth_fallback = ChatGroq(model=synth_fallback_model, api_key=api_key, temperature=0.3, max_tokens=2048, max_retries=1)
                try:
                    final_response = await llm_synth_fallback.ainvoke(synthesis_messages)
                except Exception as e2:
                    logger.warning("[agent_node] Synthesis fallback also failed: %s — using raw tool context", e2)
                    final_response = AIMessage(content=combined_tool_context)
            else:
                logger.warning("[agent_node] Synthesis error: %s — falling back to raw tool context", e)
                final_response = AIMessage(content=combined_tool_context)


        if hasattr(final_response, "usage_metadata") and final_response.usage_metadata:
            total_tokens = final_response.usage_metadata.get("output_tokens", 0) or final_response.usage_metadata.get("total_tokens", 0)
        elif hasattr(final_response, "response_metadata") and final_response.response_metadata.get("token_usage"):
            total_tokens = final_response.response_metadata["token_usage"].get("completion_tokens", 0)

        langchain_messages.append(final_response)

    final_response = langchain_messages[-1]
    raw_ans = final_response.content if hasattr(final_response, "content") else ""
    clean_raw = re.sub(r"<tool_call>.*?</tool_call>", "", str(raw_ans), flags=re.DOTALL).strip()
    answer = _clean_llm_response(clean_raw)
    if not answer:
        answer = "Here are the retrieved details based on the latest information."
    await _emit_step(state, "Completion Node", f"✅ Response generated successfully ({len(answer)} chars).", logs)

    return {"final_answer": answer, "route_used": route_used, "sources": sources, "logs": logs, "total_tokens": total_tokens, "tools_called": tools_called}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Farmer Node (Specialized Agriculture Assistant)
# ══════════════════════════════════════════════════════════════════════════════

async def farmer_node(state: UniversalAgentState) -> dict:
    """
    Dedicated node for Farmer Mode.
    Handles weather-aware farming advice, crop market prices, and govt schemes.
    """
    query = state["query"]
    logs: list[str] = []
    sources = state.get("sources", [])
    total_tokens = state.get("total_tokens", 0)
    tools_called = state.get("tools_called", [])
    await _emit_step(state, "Farmer Node", f"🌾 Processing agricultural query: '{query[:50]}'...", logs)
    llm_fast, llm_smart = _get_llms(state.get("user_groq_key", ""), state.get("agent_model", ""))

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %A")
    
    # Check if it's a weather query
    if is_weather_query(query):
        location = _extract_weather_location(query)
        if not location:
            location = "your region"
            
        await _emit_step(state, "Tools Execution Node", f"🌦️ Calling: tool_get_weather(location='{location}')", logs)
        tools_called.append("tool_get_weather")
        weather_data = await get_weather(location)
        formatted_weather = format_weather_result(weather_data)
        
        await _emit_step(state, "Synthesis Node", "⚡ Generating weather-aware agricultural advisory...", logs)
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
        await _emit_step(state, "Farmer Node", "Fetching AGMARKNET mandi prices...", logs)
        
        # Extract commodity and state using fast LLM
        resp = await llm_fast.ainvoke([HumanMessage(
            content=f"Extract the primary crop/commodity name AND the Indian state from this query. Correct any spelling (e.g. 'potatos' -> 'Potato'). Expand state abbreviations (e.g. 'UP' -> 'Uttar Pradesh'). Reply in format 'Commodity|State'. If state is unknown, reply 'Commodity|'. If commodity is unknown, reply 'Unknown|'.\nQuery: {query}"
        )])
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            total_tokens += resp.usage_metadata.get("total_tokens", 0)
        parts = resp.content.strip().split('|')
        commodity = parts[0].strip().title() if len(parts) > 0 else "Unknown"
        state_filter = parts[1].strip().title() if len(parts) > 1 else ""
        
        if commodity and commodity != "Unknown":
            tools_called.append("tool_get_mandi_prices")
            await _emit_step(state, "Tools Execution Node", f"💰 Calling: tool_get_mandi_prices(crop='{commodity}', state='{state_filter}')", logs)
            mandi_data = await get_mandi_prices(commodity, state=state_filter, max_results=10)
            
            # If AGMARKNET fails or returns nothing, fallback to Web Search
            if "error" in mandi_data or not mandi_data.get("records"):
                await _emit_step(state, "Tools Execution Node", f"🌐 AGMARKNET empty, fallback to Web Search: {commodity} mandi price", logs)
                enhanced_query = f"{commodity} price in {state_filter if state_filter else 'India'} mandi today"
                tools_called.append("tool_web_search")
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
            clean_query = re.sub(r"^(please\s+|tell me\s+|tell\s+|what is\s+|find\s+|search for\s+|about\s+|can you\s+)+", "", query, flags=re.IGNORECASE).strip()
            enhanced_query = f"{clean_query} agriculture India latest"
            await _emit_step(state, "Tools Execution Node", f"🌐 Calling: tool_web_search(query='{enhanced_query}')", logs)
            tools_called.append("tool_web_search")
            raw_results = await web_search(enhanced_query, max_results=5)
            formatted_data = format_search_results(raw_results)
            sources.extend(raw_results)
            data_context = f"LIVE WEB SEARCH RESULTS:\n{formatted_data}"
            
        await _emit_step(state, "Synthesis Node", "⚡ Formatting mandi price rates & trends...", logs)
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
        clean_query = re.sub(r"^(please\s+|tell me\s+|tell\s+|what is\s+|find\s+|search for\s+|about\s+|can you\s+)+", "", query, flags=re.IGNORECASE).strip()
        enhanced_query = f"{clean_query} agriculture India latest"
        await _emit_step(state, "Tools Execution Node", f"🌐 Calling: tool_web_search(query='{enhanced_query}')", logs)
        tools_called.append("tool_web_search")
        raw_results = await web_search(enhanced_query, max_results=5)
        formatted_web = format_search_results(raw_results)
        
        await _emit_step(state, "Synthesis Node", "⚡ Summarizing agricultural government scheme guidelines...", logs)
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
        await _emit_step(state, "Synthesis Node", "🌱 Formulating crop management & agronomy guidance...", logs)
        system = (
            "You are a specialized Agricultural Assistant for farmers in India. "
            "Answer the farmer's question using your basic knowledge. "
            "Focus on crop health, soil management, fertilizers, and best practices. "
            "Be respectful, practical, and use simple language. "
        )
        user_prompt = f"FARMER QUESTION: {query}"

    # Build hybrid memory (precise long-term summary of older turns + last 6 verbatim messages)
    long_term_summary, langchain_history = _build_hybrid_history(
        state.get("messages", []),
        state.get("uploaded_files", [])
    )
    if long_term_summary:
        system = system + f"\n{long_term_summary}"

    messages_to_send = [SystemMessage(content=system)] + langchain_history + [HumanMessage(content=user_prompt)]

    final_response = await llm_smart.ainvoke(messages_to_send)
    if hasattr(final_response, "usage_metadata") and final_response.usage_metadata:
        total_tokens = final_response.usage_metadata.get("output_tokens", 0) or final_response.usage_metadata.get("total_tokens", 0)
    elif hasattr(final_response, "response_metadata") and final_response.response_metadata.get("token_usage"):
        total_tokens = final_response.response_metadata["token_usage"].get("completion_tokens", 0)
    answer = _clean_llm_response(final_response.content)
    await _emit_step(state, "Completion Node", "✅ Farmer advisory response ready.", logs)
    
    return {
        "final_answer": answer,
        "route_used": "farmer",
        "sources": sources,
        "logs": logs,
        "total_tokens": total_tokens,
        "tools_called": tools_called
    }
