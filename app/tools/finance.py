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


# ── Cricket Scores (free RapidAPI-style endpoint) ─────────────────────────────

def _fetch_cricket_sync() -> dict[str, Any]:
    """
    Fetch live cricket scores using the free cricbuzz-cricket API on RapidAPI.
    Falls back to a web-scrape approach if no key is available.
    """
    try:
        # Using publicly available cricket score endpoint (no auth needed)
        url = "https://api.cricapi.com/v1/currentMatches?apikey=demo&offset=0"
        resp = req_lib.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            matches = data.get("data", [])[:5]  # top 5 matches
            return {"matches": matches, "source": "cricapi.com"}
    except Exception as exc:
        logger.warning("[finance] cricapi error: %s", exc)

    # Fallback: DuckDuckGo search for live scores
    return {"matches": [], "note": "Live API unavailable. Try asking: 'search for live cricket scores today'"}


async def get_cricket_scores() -> dict[str, Any]:
    """Async live cricket scores."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_cricket_sync)


def format_cricket_result(data: dict) -> str:
    """Format cricket data as markdown."""
    matches = data.get("matches", [])
    if not matches:
        return f"⚠️ {data.get('note', 'No live matches found.')}"

    lines = ["## 🏏 Live Cricket Scores\n"]
    for m in matches:
        lines.append(f"**{m.get('name', 'Match')}**")
        lines.append(f"Status: {m.get('status', 'N/A')}")
        for team in m.get("score", []):
            lines.append(f"  - {team.get('inning', '')}: {team.get('r', 0)}/{team.get('w', 0)} ({team.get('o', 0)} ov)")
        lines.append("")
    return "\n".join(lines)
