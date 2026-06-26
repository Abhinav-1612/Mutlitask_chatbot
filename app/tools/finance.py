"""
app/tools/finance.py — Finance & Live Data Tools
=================================================
• yfinance  → stock prices, info, history (free)
• requests  → live cricket scores via cricbuzz API (free tier)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests as req_lib

logger = logging.getLogger(__name__)


# ── Stock Prices (yfinance) ────────────────────────────────────────────────────

def _fetch_stock_sync(ticker: str) -> dict[str, Any]:
    """
    Fetch stock data directly from Yahoo Finance v8 chart API.
    More reliable than yfinance which breaks when Yahoo rate-limits scraping.
    """
    import json as _json
    import urllib.request as _req

    ticker = ticker.upper()
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
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
            return {"ticker": ticker, "error": "No data returned from Yahoo Finance."}

        meta = result[0].get("meta", {})
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        last_close = round(float(closes[-1]), 2) if closes else None

        price = (
            meta.get("regularMarketPrice")
            or meta.get("chartPreviousClose")
            or last_close
        )
        if price is None:
            return {"ticker": ticker, "error": "Price not available. Market may be closed."}

        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        change_pct = 0.0
        if prev_close and price:
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        return {
            "ticker":     ticker,
            "company":    meta.get("longName") or meta.get("shortName") or ticker,
            "price":      round(float(price), 2),
            "currency":   meta.get("currency", "USD"),
            "change_pct": change_pct,
            "market_cap": None,   # not in chart API
            "52w_high":   meta.get("fiftyTwoWeekHigh"),
            "52w_low":    meta.get("fiftyTwoWeekLow"),
            "sector":     "N/A",  # not in chart API
        }

    except Exception as exc:
        logger.error("[finance] Yahoo chart API error for %s: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc)}


def _fetch_stock_sync_fallback(ticker: str) -> dict[str, Any]:
    """Try yfinance as a secondary fallback."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return {"ticker": ticker, "error": "No historical data from yfinance."}
        price = round(float(hist["Close"].iloc[-1]), 2)
        return {"ticker": ticker, "company": ticker, "price": price,
                "currency": "USD", "change_pct": 0.0, "market_cap": None,
                "52w_high": None, "52w_low": None, "sector": "N/A"}
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}




async def get_stock_price(ticker: str) -> dict[str, Any]:
    """Async stock price fetch."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_stock_sync, ticker)
    logger.info("[finance] Stock %s → %s", ticker, result.get("price"))
    return result


def format_stock_result(data: dict) -> str:
    """Format stock data as markdown."""
    if "error" in data:
        return f"❌ Could not fetch data for **{data['ticker']}**: {data['error']}"
    chg_emoji = "📈" if data.get("change_pct", 0) >= 0 else "📉"
    return (
        f"## {chg_emoji} {data['company']} ({data['ticker']})\n"
        f"- **Price**: {data['currency']} {data['price']}\n"
        f"- **Change**: {data['change_pct']:+.2f}%\n"
        f"- **52W High/Low**: {data['52w_high']} / {data['52w_low']}\n"
        f"- **Sector**: {data['sector']}\n"
    )


# ── Cricket Scores (multi-strategy with fallbacks) ───────────────────────────

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


def _fetch_cricket_espn() -> dict:
    """Strategy 3: ESPNcricinfo consumer API — live scorecard data."""
    import json as _json
    import urllib.request as _ureq

    url = "https://hs-consumer-api.espncricinfo.com/v1/pages/livescores"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept": "application/json",
        "Origin": "https://www.espncricinfo.com",
        "Referer": "https://www.espncricinfo.com/",
    }
    req = _ureq.Request(url, headers=headers)
    with _ureq.urlopen(req, timeout=12) as r:
        content = r.read().decode("utf-8")

    data = _json.loads(content)
    # Structure: {"content": [{"matches": [...]}]}
    content_list = data.get("content", [])
    matches = []
    for section in content_list:
        for m in section.get("matches", []):
            info = m.get("matchInfo", {})
            score_info = m.get("matchScore", {})

            team1 = info.get("team1", {}).get("teamName", "")
            team2 = info.get("team2", {}).get("teamName", "")
            status = info.get("status", "")
            series = info.get("seriesName", "")

            scores = []
            for team_key, team_name in (("team1Score", team1), ("team2Score", team2)):
                ts = score_info.get(team_key) or {}
                for inn_key in ("inngs1", "inngs2"):
                    inn = ts.get(inn_key)
                    if inn:
                        scores.append({
                            "inning": team_name,
                            "r": inn.get("runs", 0),
                            "w": inn.get("wickets", 0),
                            "o": inn.get("overs", 0),
                        })

            if team1 and team2:
                matches.append({
                    "name": f"{team1} vs {team2}",
                    "series": series,
                    "status": status,
                    "score": scores,
                })
            if len(matches) >= 6:
                break
        if len(matches) >= 6:
            break

    return {"matches": matches, "source": "ESPN Cricinfo"}


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


def _fetch_cricket_sync(query: str = "") -> dict[str, Any]:
    """
    Fetch live cricket scores using a cascading strategy:
      1. RapidAPI Cricbuzz (free tier, if RAPIDAPI_KEY set)
      2. cricapi.com   (if CRICAPI_KEY set)
      3. ESPN Cricinfo (web scrape, no auth)
      4. DuckDuckGo news search (final fallback)
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

    # Strategy 3 — ESPN Cricinfo scrape
    try:
        result = _fetch_cricket_espn()
        if result.get("matches"):
            logger.info("[finance] Cricket via ESPN Cricinfo ✓")
            return result
    except Exception as exc:
        logger.warning("[finance] ESPN Cricinfo failed: %s", exc)

    # Strategy 4 — DuckDuckGo news fallback
    logger.info("[finance] Cricket via DuckDuckGo fallback")
    return _fetch_cricket_ddg_fallback(query)


async def get_cricket_scores(query: str = "") -> dict[str, Any]:
    """Async live cricket scores."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_cricket_sync, query)


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

