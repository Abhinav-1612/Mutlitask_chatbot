
#"This file provides financial and live sports tools. 
# It fetches stock prices from Yahoo Finance and live cricket scores using 
# multiple APIs with fallback mechanisms."

"""
app/tools/finance.py — Finance & Live Data Tools
=================================================
• yfinance  → stock prices, info, history (free)
• requests  → live cricket scores via cricbuzz API (free tier)
"""
from __future__ import annotations

import asyncio #Prevent FastAPI from blocking while waiting for the API response.
import logging #For logger and debuging .
from typing import Any

import requests as req_lib#for making api calls.

logger = logging.getLogger(__name__)


# ── Stock Prices (yfinance) ────────────────────────────────────────────────────
# Gets live stock price from Yahoo Finance API.
# Returns company name, price, currency, 52-week high/low, etc.
def _fetch_stock_sync(ticker: str) -> dict[str, Any]:
    """
    Fetch stock data directly from Yahoo Finance v8 chart API.
    More reliable than yfinance which breaks when Yahoo rate-limits scraping.
    """
    import json as _json
    import urllib.request as _req
    import re as _re

    ticker_clean = _re.sub(r"[^A-Za-z0-9.\-^]", "", str(ticker).strip().upper())
    if not ticker_clean:
        return {"ticker": "UNKNOWN", "error": "Invalid or empty stock ticker symbol."}

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_clean}"
        f"?interval=1d&range=5d"
    )
    try:
        request = _req.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        )
        with _req.urlopen(request, timeout=10) as response:
            data = _json.loads(response.read().decode("utf-8"))

        result = data.get("chart", {}).get("result") or []
        if not result:
            fb = _fetch_stock_sync_fallback(ticker_clean)
            if "error" not in fb:
                return fb
            return {"ticker": ticker_clean, "error": "No data returned from Yahoo Finance."}

        meta = result[0].get("meta", {})
        indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
        closes = indicators.get("close", []) or []
        valid_closes = [float(c) for c in closes if c is not None]
        last_close = round(valid_closes[-1], 2) if valid_closes else None

        raw_price = (
            meta.get("regularMarketPrice")
            or meta.get("chartPreviousClose")
            or last_close
        )
        if raw_price is None:
            fb = _fetch_stock_sync_fallback(ticker_clean)
            if "error" not in fb:
                return fb
            return {"ticker": ticker_clean, "error": "Price not available. Market may be closed."}

        price = round(float(raw_price), 2)
        raw_prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        prev_close = float(raw_prev) if raw_prev is not None else None
        change_pct = 0.0
        if prev_close and price:
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        return {
            "ticker":     ticker_clean,
            "company":    meta.get("longName") or meta.get("shortName") or ticker_clean,
            "price":      price,
            "prev_close": prev_close,
            "currency":   meta.get("currency", "USD"),
            "change_pct": change_pct,
            "market_cap": None,   # not in chart API
            "52w_high":   meta.get("fiftyTwoWeekHigh"),
            "52w_low":    meta.get("fiftyTwoWeekLow"),
            "sector":     "N/A",  # not in chart API
        }

    except Exception as exc:
        logger.error("[finance] Yahoo chart API error for %s: %s", ticker_clean, exc)
        fb = _fetch_stock_sync_fallback(ticker_clean)
        if "error" not in fb:
            return fb
        return {"ticker": ticker_clean, "error": str(exc)}

# if yahhoo api fails then uses yfinance library 
def _fetch_stock_sync_fallback(ticker: str) -> dict[str, Any]:
    """Try yfinance as a secondary fallback."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return {"ticker": ticker, "error": "No historical data from yfinance."}
        price = round(float(hist["Close"].iloc[-1]), 2)
        return {"ticker": ticker, "company": ticker, "price": price,
                "prev_close": None, "currency": "USD", "change_pct": 0.0, "market_cap": None,
                "52w_high": None, "52w_low": None, "sector": "N/A"}
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}



#async version runs stock fetch without blocking fastapi 
async def get_stock_price(ticker: str) -> dict[str, Any]:
    """Async stock price fetch."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_stock_sync, ticker)
    logger.info("[finance] Stock %s → %s", ticker, result.get("price"))
    return result

# convert stck result from json to markdown for user  
def format_stock_result(data: dict) -> str:
    """Format stock data as markdown."""
    if "error" in data:
        return f"❌ Could not fetch data for **{data['ticker']}**: {data['error']}"
    chg_emoji = "📈" if data.get("change_pct", 0) >= 0 else "📉"
    prev_close_val = data.get('prev_close')
    prev_close_str = f"{data['currency']} {prev_close_val}" if prev_close_val is not None else "N/A"
    return (
        f"## {chg_emoji} {data['company']} ({data['ticker']})\n"
        f"- **Current Price**: {data['currency']} {data['price']}\n"
        f"- **Yesterday's Closing Price (Previous Close)**: {prev_close_str}\n"
        f"- **Change**: {data['change_pct']:+.2f}%\n"
        f"- **52W High/Low**: {data['52w_high']} / {data['52w_low']}\n"
        f"- **Sector**: {data['sector']}\n"
    )


# ── Cricket Scores (multi-strategy with fallbacks) ───────────────────────────
# fectch live cricket score using rapidapi 
def _fetch_cricket_rapidapi(rapidapi_key: str) -> dict:
    """Strategy 1: Cricbuzz via RapidAPI (free tier, best data quality)."""
    url = "https://cricbuzz-cricket.p.rapidapi.com/matches/v1/live"
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com",
    }
    resp = req_lib.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    matches = []
    for type_match in data.get("typeMatches", []):
        for series in type_match.get("seriesMatches", []):
            series_wrapper = series.get("seriesAdWrapper") or {}
            for match in series_wrapper.get("matches", []):
                mi = match.get("matchInfo", {})
                ms = match.get("matchScore", {})

                team1 = mi.get("team1", {}).get("teamName", "Team 1")
                team2 = mi.get("team2", {}).get("teamName", "Team 2")
                status = mi.get("status", "")
                match_desc = mi.get("matchDesc", "")
                series_name = mi.get("seriesName", "")

                scores = []
                for team_key in ("team1Score", "team2Score"):
                    ts = ms.get(team_key) or {}
                    for inning_key in ("inngs1", "inngs2"):
                        inn = ts.get(inning_key)
                        if inn:
                            scores.append({
                                "inning": f"{team1 if team_key == 'team1Score' else team2}",
                                "r": inn.get("runs", 0),
                                "w": inn.get("wickets", 0),
                                "o": inn.get("overs", 0),
                            })

                matches.append({
                    "name": f"{team1} vs {team2}",
                    "series": series_name,
                    "matchDesc": match_desc,
                    "status": status,
                    "score": scores,
                })
                if len(matches) >= 6:
                    break

    return {"matches": matches, "source": "Cricbuzz via RapidAPI"}

# if rapid api fails then uses crickapi 
def _fetch_cricket_cricapi(cricapi_key: str) -> dict:
    """Strategy 2: cricapi.com with a real user API key."""
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={cricapi_key}&offset=0"
    resp = req_lib.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise ValueError(f"cricapi error: {data.get('info', 'unknown')}")
    matches = data.get("data", [])[:6]
    return {"matches": matches, "source": "cricapi.com"}

# Strategy 3: Cricinfo RSS for Real-Time Ball-by-Ball Live Scores
def _fetch_cricket_cricinfo_rss() -> dict:
    """Strategy 3: Cricinfo Live Scores RSS — highly reliable realtime ball-by-ball updates."""
    import urllib.request as _ureq
    import xml.etree.ElementTree as ET
    import re

    url = "https://static.cricinfo.com/rss/livescores.xml"
    req = _ureq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with _ureq.urlopen(req, timeout=10) as r:
        content = r.read().decode("utf-8")
        
    root = ET.fromstring(content)
    matches = []
    
    for item in root.findall(".//item"):
        title = (item.find("title").text or "").strip()
        link = (item.find("link").text or "").strip()
        guid = (item.find("guid").text or "").strip()
        
        # Titles are usually "TeamA 123/4 * v TeamB 456/10"
        parts = re.split(r'\s+v\s+', title, maxsplit=1)
        if len(parts) == 2:
            team1_str = parts[0].strip()
            team2_str = parts[1].strip()
            name = f"{team1_str} vs {team2_str}"
        else:
            name = title
            
        status = "Live match in progress" if "*" in title else "Match update"
        
        matches.append({
            "name": name,
            "status": status,
            "score": [], 
            "url": guid or link
        })
        
        if len(matches) >= 8:
            break

    return {"matches": matches, "source": "Cricinfo Live RSS"}

#if everything fails then uses duckduck go 
def _fetch_cricket_ddg_fallback(query: str = "") -> dict:
    """Strategy 4: DuckDuckGo search — richer cricket score summaries."""
    try:
        from duckduckgo_search import DDGS
        summaries: list[dict] = []
        
        # Use the specific user query if provided, otherwise fallback to generic
        search_queries = [query] if query else [
            "live cricket score today 2024",
            "cricket match score right now",
        ]
        with DDGS() as ddgs:
            for sq in search_queries:
                try:
                    results = list(ddgs.news(
                        sq,
                        region="in-en",        # India region for better cricket coverage
                        safesearch="moderate",
                        timelimit="d",
                        max_results=5,
                    ))
                    for r in results:
                        title = r.get("title", "Match")
                        body  = r.get("body", "")[:300]
                        url   = r.get("url", "")
                        # Deduplicate by title
                        if not any(s["name"] == title for s in summaries):
                            summaries.append({
                                "name": title,
                                "status": body,
                                "score": [],
                                "url": url,
                            })
                    if len(summaries) >= 6:
                        break
                except Exception:
                    continue

        if summaries:
            return {"matches": summaries[:6], "source": "DuckDuckGo News", "is_text": True}
        return {"matches": [], "note": "Could not fetch live cricket scores right now. Please try again shortly."}
    except Exception:
        return {"matches": [], "note": "Could not fetch live cricket scores. Please try again."}

# if DDG fails, try Tavily Search
def _fetch_cricket_tavily_fallback(query: str = "") -> dict:
    """Strategy 5: Tavily Search — fallback when DDG/ESPN block us."""
    from app.config import settings
    if not settings.tavily_api_key:
        return {"matches": [], "note": "Could not fetch live cricket scores right now (no Tavily API key)."}

    try:
        import requests as req_lib
        api_key = settings.tavily_api_key.strip()
        url = "https://api.tavily.com/search"
        
        search_query = query.strip()
        if "score" not in search_query.lower() and "match" not in search_query.lower():
            search_query += " live cricket score today match"
            
        payload = {
            "api_key": api_key,
            "query": search_query or "live cricket score today match",
            "topic": "news",
            "days": 1,
            "search_depth": "advanced",
            "include_answer": False,
            "include_images": False,
            "max_results": 8,
        }
        resp = req_lib.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results", [])
        
        if not raw_results:
            # Fallback to 2 days for timezone coverage
            payload["days"] = 2
            resp2 = req_lib.post(url, json=payload, timeout=10)
            if resp2.status_code == 200:
                raw_results = resp2.json().get("results", [])

        cricket_keywords = ("cricket", "test", "score", "innings", "stumps", "runs", "wickets", "overs", "ind", "sl", "india", "sri lanka", "cricinfo", "cricbuzz", "bcci", "icc", "ball", "batting", "bowling")
        unrelated_sports = ("hockey", "football", "soccer", "tennis", "badminton", "formula 1", "nba", "basketball", "wrestling", "olympics")

        summaries = []
        for r in raw_results:
            title = r.get("title", "Match Update")
            body = r.get("content", "")[:400]
            url = r.get("url", "")
            combined_text = f"{title.lower()} {body.lower()}"
            
            # Skip if explicitly about a different sport
            if any(un_sport in title.lower() for un_sport in unrelated_sports):
                continue
                
            # Must have at least one cricket indicator
            if not any(kw in combined_text for kw in cricket_keywords):
                continue

            if not any(s["name"] == title for s in summaries):
                summaries.append({
                    "name": title,
                    "status": body,
                    "score": [],
                    "url": url,
                })
        if summaries:
            return {"matches": summaries, "source": "Tavily News", "is_text": True}
        return {"matches": [], "note": "Could not fetch live cricket scores right now. Please try again shortly."}
    except Exception as exc:
        logger.warning("[finance] Tavily cricket fallback failed: %s", exc)
        return {"matches": [], "note": f"Could not fetch live cricket scores (Tavily error: {exc})."}

# this function tries every cricket api from start whichevr eorks return its data 
def _fetch_cricket_sync(query: str = "") -> dict[str, Any]:
    """
    Fetch live cricket scores using a cascading strategy:
      1. RapidAPI Cricbuzz (free tier, if RAPIDAPI_KEY set)
      2. cricapi.com   (if CRICAPI_KEY set)
      3. Cricinfo Live RSS (real-time ball-by-ball, no auth)
      4. DuckDuckGo news search (final fallback)
      5. Tavily Search (reliable backup fallback)
    """
    from app.config import settings

    # Strategy 1 — RapidAPI Cricbuzz
    if settings.rapidapi_key:
        try:
            result = _fetch_cricket_rapidapi(settings.rapidapi_key)
            if result.get("matches"):
                logger.info("[finance] Cricket via RapidAPI Cricbuzz ✓")
                return result
        except Exception as exc:
            logger.warning("[finance] RapidAPI Cricbuzz failed: %s", exc)

    # Strategy 2 — cricapi.com with real key
    if settings.cricapi_key:
        try:
            result = _fetch_cricket_cricapi(settings.cricapi_key)
            if result.get("matches"):
                logger.info("[finance] Cricket via cricapi.com ✓")
                return result
        except Exception as exc:
            logger.warning("[finance] cricapi.com failed: %s", exc)

    # Strategy 3 — Cricinfo Live RSS (Fastest and most reliable)
    try:
        result = _fetch_cricket_cricinfo_rss()
        if result.get("matches"):
            logger.info("[finance] Cricket via Cricinfo RSS ✓")
            return result
    except Exception as exc:
        logger.warning("[finance] Cricinfo RSS failed: %s", exc)

    # Strategy 4 — DuckDuckGo news fallback
    logger.info("[finance] Cricket via DuckDuckGo fallback")
    ddg_result = _fetch_cricket_ddg_fallback(query)
    if ddg_result.get("matches"):
        return ddg_result

    # Strategy 5 — Tavily Search fallback
    logger.info("[finance] Cricket via Tavily fallback")
    return _fetch_cricket_tavily_fallback(query)


async def get_cricket_scores(query: str = "") -> dict[str, Any]:
    """Async live cricket scores."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_cricket_sync, query)

# convert result from json to markdown 
def format_cricket_result(data: dict) -> str:
    """Format cricket data as a rich Markdown scorecard."""
    matches = data.get("matches", [])
    source = data.get("source", "Live API")
    is_text = data.get("is_text", False)   # DuckDuckGo fallback returns plain text

    if not matches:
        return f"⚠️ {data.get('note', 'No live matches found right now. Try again in a moment.')}"

    lines = [f"## 🏏 Live Cricket Scores\n*Source: {source}*\n"]

    for m in matches:
        name   = m.get("name", "Match")
        status = m.get("status", "")
        series = m.get("series", "")
        scores = m.get("score", [])
        url    = m.get("url", "")

        header = f"### {name}"
        if series:
            header += f" · *{series}*"
        lines.append(header)

        if is_text:
            # DuckDuckGo fallback — status IS the news snippet
            lines.append(f"> {status}")
            if url:
                lines.append(f"[🔗 Read more]({url})")
        else:
            if scores:
                for sc in scores:
                    overs = sc.get("o", 0)
                    lines.append(
                        f"  🏏 **{sc.get('inning','?')}**: "
                        f"{sc.get('r', 0)}/{sc.get('w', 0)}"
                        f"{f' ({overs} ov)' if overs else ''}"
                    )
            if status:
                lines.append(f"  📊 *{status}*")

        lines.append("")   # blank line between matches

    return "\n".join(lines).rstrip()

