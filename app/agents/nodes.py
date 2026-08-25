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

def _get_llms(user_key: str = "", agent_model: str = "") -> tuple:
    """
    Return (llm_fast, llm_smart) for the given API key and model.
    Instances are cached per key+model so we don’t recreate on every token.
    Falls back to the server’s GROQ_API_KEY if user_key is empty.
    """
    api_key = (user_key.strip() or settings.groq_api_key)
    model_name = (agent_model.strip() or settings.agent_model)
    cache_key = f"{api_key}_{model_name}"

    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = (
            ChatGroq(
                model=settings.router_model,
                api_key=api_key,
                temperature=0.0,
                max_tokens=512,
            ),
            ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0.3,
                max_tokens=2048,
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
        results = await news_search(topic_or_location, max_results=6, freshness="d")
        return format_news_results(results, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    except Exception as e:
        return f"News Error: {str(e)}"

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    news_intent: bool = Field(default=False, description="Set to true if looking for recent news or latest updates.")

@tool(args_schema=WebSearchInput)
async def tool_web_search(query: str, news_intent: bool = False) -> str:
    """Search the web for general knowledge, information, websites, or current events."""
    try:
        if news_intent or is_news_query(query):
            results = await news_search(query, max_results=6, freshness="d")
            return format_news_results(results, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        else:
            results = await web_search(query, max_results=6)
            return format_search_results(results) if results else "No results found."
    except Exception as e:
        return f"Search Error: {str(e)}"

class StockInput(BaseModel):
    ticker_or_company: str = Field(description="The stock ticker symbol or company name. Example: 'TSLA', 'Tesla', 'AAPL', or 'RELIANCE.NS'.")

@tool(args_schema=StockInput)
async def tool_get_stock_price(ticker_or_company: str) -> str:
    """Fetch live and historical stock data (current price, yesterday's exact closing price / previous close, change %, 52W range) for any company or ticker symbol."""
    try:
        from app.agents.nodes import _extract_clean_ticker
        clean_ticker = _extract_clean_ticker(ticker_or_company, ticker_or_company) or ticker_or_company
        data = await get_stock_price(clean_ticker)
        return format_stock_result(data)
    except Exception as e:
        return f"Finance Error: {str(e)}"

class CricketInput(BaseModel):
    query: str = Field(description="The cricket match query (e.g., 'India live score', 'IPL today').")

@tool(args_schema=CricketInput)
async def tool_get_cricket_scores(query: str) -> str:
    """Fetch live cricket scores and match updates."""
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

class MandiInput(BaseModel):
    commodity: str = Field(description="The name of the crop or commodity (e.g., 'Soyabean', 'Wheat').")
    state: str = Field(default="", description="The Indian state (e.g., 'Maharashtra'). Optional.")

@tool(args_schema=MandiInput)
async def tool_get_mandi_prices(commodity: str, state: str = "") -> str:
    """Fetch live crop market (mandi) prices from AGMARKNET."""
    try:
        data = await get_mandi_prices(commodity, state=state, max_results=10)
        return format_mandi_prices(data, commodity)
    except Exception as e:
        return f"Mandi Error: {str(e)}"

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


def _clean_llm_response(text: str) -> str:
    """Remove internal reasoning or thinking process artifacts from LLM response."""
    if not text:
        return ""
    # 1. Remove <think>...</think> tags (case-insensitive)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Remove any loose or unclosed <think> / </think> tags
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    
    # 3. Strip thinking process text if present
    if re.search(r"^(?:Here['’]?s (?:a )?thinking process|Thinking Process):", text, flags=re.IGNORECASE):
        patterns = [
            r"(?:\[Output Generation\]|Outputs? response|Final Answer:?|Output:?)\s*",
            r"(?=\n\n(?:###|##|#|\*\*|Today|India|Sri Lanka|Match|🏏|📈))",
        ]
        for pat in patterns:
            parts = re.split(pat, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[-1]
                break
        else:
            match = re.search(r"(\n(?:###|##|#|\*\*|Today|India|Sri Lanka|Match|🏏|📈).*)", text, flags=re.DOTALL)
            if match:
                text = match.group(1)
                
    # 4. Remove search/citation artifacts like 【1†L1-L3】
    text = re.sub(r"【.*?】", "", text)
                
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
    # 1. First check if query directly mentions a known company
    if query:
        q_upper = query.upper()
        for comp in sorted(COMMON_TICKER_MAP.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(comp)}\b", q_upper):
                return COMMON_TICKER_MAP[comp]

    if not raw_text:
        return ""

    # 2. Clean thinking tags and reasoning artifacts
    cleaned = _clean_llm_response(raw_text)

    # 3. Check cleaned text against known company names
    c_upper = cleaned.upper()
    for comp in sorted(COMMON_TICKER_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(comp)}\b", c_upper):
            return COMMON_TICKER_MAP[comp]

    # 4. Remove markdown / formatting noise
    cleaned = re.sub(r"```[\w]*", "", cleaned)
    cleaned = re.sub(r"[`\"'*_#:]", " ", cleaned)

    # 5. Extract words and look for ticker candidates (e.g. AAPL, TSLA, RELIANCE.NS, BTC-USD, ^NSEI)
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

    # 6. Fallback: extract uppercase ticker candidates directly from the query itself
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
# NODE 1 — Omni Agent Node
# ══════════════════════════════════════════════════════════════════════════════

async def agent_node(state: UniversalAgentState) -> dict:
    """
    Omni-Agent that natively binds all tools and executes them in parallel if needed.
    """
    query = state["query"]
    logs  = [_ts("agent_node", "Processing query...")]
    _, llm_smart = _get_llms(state.get("user_groq_key", ""), state.get("agent_model", ""))
    
    uploaded = state.get("uploaded_files", [])
    
    # ── Dynamically define Qdrant tool if files are uploaded
    @tool
    async def tool_search_document(search_query: str) -> str:
        """Search the currently uploaded PDF document for context."""
        if not uploaded:
            return "No document is currently uploaded."
        try:
            file_ids = [f["file_id"] for f in uploaded if "file_id" in f]
            filter_by = {"file_id": file_ids[0]} if file_ids else None
            hits = similarity_search(search_query, top_k=5, filter_payload=filter_by)
            if not hits:
                return "No relevant information found in the document."
            return "\n\n---\n\n".join(hit["content"] for hit in hits)
        except Exception as e:
            return f"Document Search Error: {str(e)}"

    _tools = [
        tool_get_weather, tool_search_news, tool_web_search, tool_get_stock_price, 
        tool_get_cricket_scores, tool_search_arxiv, tool_get_mandi_prices,
        tool_search_document
    ]
    llm_with_tools = llm_smart.bind_tools(_tools)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    system_prompt = (
        "You are Omni-Agent, a top-tier AI assistant with real-time tool access.\n"
        f"Today's date and time (UTC): {now}.\n\n"
        "MANDATORY MULTI-INTENT TOOL CALLING RULES:\n"
        "1. Check the user's prompt for ALL questions and intents. If the user asks about multiple topics (e.g. stock price, weather, news), you MUST call ALL corresponding tools in parallel in the FIRST turn!\n"
        "2. FOR NEWS / HEADLINES / CURRENT EVENTS / BREAKING UPDATES: ALWAYS call `tool_search_news`. DO NOT call web search for news.\n"
        "3. FOR STOCKS / TICKERS / MARKET / CLOSING PRICES: ALWAYS call `tool_get_stock_price`. DO NOT call web search for stocks.\n"
        "4. FOR WEATHER / TEMPERATURE / FORECAST: ALWAYS call `tool_get_weather`. DO NOT call web search for weather.\n"
        "5. FOR CRICKET / LIVE SCORES: ALWAYS call `tool_get_cricket_scores`.\n"
        "6. FOR GENERAL WEBSITES / SEARCH: Call `tool_web_search`.\n"
        "7. FOR UPLOADED DOCUMENTS: Call `tool_search_document`.\n\n"
        "CRITICAL COMPLETION & SYNTHESIS RULES:\n"
        "• Assume all tool results provided in the conversation are the latest and sufficient available data. NEVER perform repeated or redundant searches for the same topic.\n"
        "• Always provide a comprehensive, complete summary covering all key facts, numbers, and takeaways.\n"
        "• DOCUMENT SUMMARIES & RESUMES: Format sections using clean Markdown subheadings (###) and bulleted lists. Avoid cramming large multiline text or multi-item lists into Markdown table cells with raw `<br>` tags.\n"
        "• CLEAN MARKDOWN ONLY: Never output raw HTML tags like `<br>`, `<span>`, `<div>`, `<b>`, `<p>` in text responses. Use clean, native Markdown formatting.\n"
        "• For news / web articles: Present all key headlines and summaries. Cite sources with clickable Markdown links using their real URLs (e.g. `[NDTV](https://...)` or `[Read Article →](https://...)`) whenever URLs are present in the tool output. If a story does not have a direct URL, summarize the news naturally without refusing to answer.\n"
        "• For stocks, report the company name, current price, yesterday's closing price (previous close), change %, and 52W range.\n"
        "• For weather, report current temperature, conditions, humidity, wind, and forecast.\n"
        "• NEVER output raw unformatted search results or raw URL dumps."
    )

    # Reconstruct history
    history = state.get("messages", [])
    langchain_messages = [SystemMessage(content=system_prompt)]
    
    for msg in history:
        if msg.get("role") == "user":
            langchain_messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            langchain_messages.append(AIMessage(content=msg.get("content", "")))
    
    langchain_messages.append(HumanMessage(content=query))
    
    route_used = "general" # Default, updated if tools are called

    # ── Tool Execution Loop ──────────────────────────────────────────────────────
    max_rounds = 3
    for round_idx in range(max_rounds):
        response = await llm_with_tools.ainvoke(langchain_messages)
        
        if not response.tool_calls:
            langchain_messages.append(response)
            break
            
        langchain_messages.append(response)
        
        tasks = []
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            logs.append(_ts("agent_node", f"Tool call: {name}({args})"))
            
            # Map tool name to function
            tool_func = None
            for t in _tools:
                if t.name == name:
                    tool_func = t
                    break
                    
            if tool_func:
                tasks.append((tc["id"], name, tool_func.ainvoke(args)))
            else:
                async def err_func(n): return f"Unknown tool: {n}"
                tasks.append((tc["id"], name, err_func(name)))
                
        # Execute all tools concurrently in parallel
        results = await asyncio.gather(*(t[2] for t in tasks), return_exceptions=True)
        
        for i, (tc_id, name, result) in enumerate(zip((t[0] for t in tasks), (t[1] for t in tasks), results)):
            if isinstance(result, Exception):
                logger.error("[agent_node] Tool %s failed: %s", name, result)
                result = f"Tool execution failed: {str(result)}"
                
            langchain_messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))
            
            # Update route_used for UI flair (prioritize document/rag before general search)
            if "document" in name or "arxiv" in name or "rag" in name or "pdf" in name:
                route_used = "rag"
            elif "stock" in name or "cricket" in name or "finance" in name:
                route_used = "finance"
            elif "weather" in name or "search" in name or "news" in name or "web" in name:
                route_used = "web"
                
    # If the loop ended after executing tools without a final text response, invoke once more for synthesis
    if isinstance(langchain_messages[-1], ToolMessage) or not getattr(langchain_messages[-1], "content", "").strip():
        langchain_messages.append(HumanMessage(content="You have gathered all needed information from tools. Now synthesize and write the complete final response addressing every part of the original request in clean Markdown. Do NOT call any more tools."))
        final_response = await llm_with_tools.ainvoke(langchain_messages)
        langchain_messages.append(final_response)

    final_response = langchain_messages[-1]
    answer = _clean_llm_response(final_response.content if hasattr(final_response, "content") else "")
    logs.append(_ts("agent_node", f"Response ready ({len(answer)} chars)."))
    
    return {"final_answer": answer, "route_used": route_used, "logs": logs}


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
    llm_fast, llm_smart = _get_llms(state.get("user_groq_key", ""), state.get("agent_model", ""))

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

    answer = _clean_llm_response(response.content)
    logs.append(_ts("farmer_node", "Farmer response complete."))
    
    return {
        "final_answer": answer,
        "sources": sources,
        "route_used": "farmer",
        "logs": logs,
    }
