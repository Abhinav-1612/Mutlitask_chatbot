"""
Fresh web/news search and structured current-weather tools.

News search waterfall (news queries only):
  1. NewsAPI (newsapi.org)  — primary, 200 req/day free tier, returns images
  2. Tavily                 — fallback when NewsAPI quota is hit
  3. Google News RSS        — last resort if Tavily also fails

General web search (all other queries):
  DuckDuckGo — unchanged from before
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)


class NewsAPIQuotaExceeded(Exception):
    """Raised when NewsAPI returns a 429 / quota-exhausted response."""


_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _fetch_json(url: str, timeout: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Omni-Agent/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_weather_sync(location: str) -> dict[str, Any]:
    location = location.strip()
    if not location:
        return {"error": "No city or location was provided."}

    try:
        geocode_query = urllib.parse.urlencode(
            {"name": location, "count": 1, "language": "en", "format": "json"}
        )
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?{geocode_query}"
        geocode_data = _fetch_json(geocode_url)
        matches = geocode_data.get("results") or []
        if not matches:
            return {"error": f"Could not find a location matching '{location}'."}

        place = matches[0]
        forecast_query = urllib.parse.urlencode(
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,rain,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": 4,
                "timezone": "auto",
            }
        )
        forecast_url = f"https://api.open-meteo.com/v1/forecast?{forecast_query}"
        forecast_data = _fetch_json(forecast_url)

        current = dict(forecast_data.get("current") or {})
        current_units = forecast_data.get("current_units") or {}
        current["condition"] = _WEATHER_CODES.get(
            current.get("weather_code"), "Unknown conditions"
        )
        current["units"] = current_units

        daily = forecast_data.get("daily") or {}
        daily_units = forecast_data.get("daily_units") or {}
        days = []
        observation_date = str(current.get("time") or "")[:10]

        def daily_value(key: str, index: int):
            values = daily.get(key) or []
            return values[index] if index < len(values) else None

        for index, date in enumerate(daily.get("time") or []):
            if observation_date and date < observation_date:
                continue
            code = daily_value("weather_code", index)
            days.append(
                {
                    "date": date,
                    "condition": _WEATHER_CODES.get(code, "Unknown conditions"),
                    "temperature_max": daily_value("temperature_2m_max", index),
                    "temperature_min": daily_value("temperature_2m_min", index),
                    "precipitation_probability_max": daily_value(
                        "precipitation_probability_max", index
                    ),
                    "units": daily_units,
                }
            )
            if len(days) == 3:
                break

        display_parts = [
            place.get("name"),
            place.get("admin1"),
            place.get("country"),
        ]
        return {
            "location": ", ".join(part for part in display_parts if part),
            "timezone": forecast_data.get("timezone"),
            "current": current,
            "forecast": days,
            "source": {
                "name": "Open-Meteo",
                "weather_url": forecast_url,
                "geocoding_url": geocode_url,
            },
        }
    except Exception as exc:
        logger.error("[weather] Failed to fetch weather for %s: %s", location, exc)
        return {"error": f"Could not fetch live weather for {location}: {exc}"}


async def get_weather(location: str) -> dict[str, Any]:
    """Get structured current weather and a three-day forecast."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_weather_sync, location)


def format_weather_result(data: dict[str, Any]) -> str:
    """Format structured Open-Meteo data as concise Markdown."""
    if data.get("error"):
        return f"Could not retrieve current weather: {data['error']}"

    current = data.get("current") or {}
    units = current.get("units") or {}
    temperature_unit = units.get("temperature_2m", "C")
    wind_unit = units.get("wind_speed_10m", "km/h")
    humidity_unit = units.get("relative_humidity_2m", "%")

    lines = [
        f"## Current weather in {data.get('location', 'the requested location')}",
        f"**Observed:** {current.get('time', 'Unknown')} ({data.get('timezone', 'local time')})",
        f"- **Conditions:** {current.get('condition', 'Unknown')}",
        f"- **Temperature:** {current.get('temperature_2m')} {temperature_unit}",
        f"- **Feels like:** {current.get('apparent_temperature')} {temperature_unit}",
        f"- **Humidity:** {current.get('relative_humidity_2m')} {humidity_unit}",
        f"- **Wind:** {current.get('wind_speed_10m')} {wind_unit}",
        f"- **Precipitation:** {current.get('precipitation')} {units.get('precipitation', 'mm')}",
    ]

    forecast = data.get("forecast") or []
    if forecast:
        lines.extend(["", "### Three-day forecast"])
        for day in forecast:
            day_units = day.get("units") or {}
            temp_unit = day_units.get("temperature_2m_max", temperature_unit)
            rain_unit = day_units.get("precipitation_probability_max", "%")
            lines.append(
                f"- **{day.get('date')}**: {day.get('condition')}, "
                f"{day.get('temperature_min')} to {day.get('temperature_max')} {temp_unit}, "
                f"rain chance {day.get('precipitation_probability_max')} {rain_unit}"
            )

    lines.extend(["", "*Live data: Open-Meteo*"])
    return "\n".join(lines)


def _clean_html(raw_html: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", raw_html or "")
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _google_news_rss_sync(
    query: str,
    max_results: int,
    freshness: str | None,
) -> list[dict[str, Any]]:
    freshness_suffix = {"d": " when:1d", "w": " when:7d", "m": " when:30d"}.get(
        freshness, ""
    )
    encoded_query = urllib.parse.quote_plus(f"{query}{freshness_suffix}")
    url = (
        "https://news.google.com/rss/search"
        f"?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Omni-Agent/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        root = ET.fromstring(response.read())

    results = []
    for item in root.findall("./channel/item")[:max_results]:
        published_raw = item.findtext("pubDate", default="")
        try:
            published_at = parsedate_to_datetime(published_raw).isoformat()
        except (TypeError, ValueError):
            published_at = published_raw
        source_node = item.find("source")
        results.append(
            {
                "title": item.findtext("title", default=""),
                "url": item.findtext("link", default=""),
                "snippet": _clean_html(item.findtext("description", default=""))[:500],
                "published_at": published_at,
                "source": source_node.text if source_node is not None else "Google News",
            }
        )
    return results


def _bing_web_rss_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "format": "rss",
            "q": query,
            "cc": "US",
            "setlang": "en-US",
        }
    )
    url = f"https://www.bing.com/search?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        root = ET.fromstring(response.read())

    results = []
    for item in root.findall("./channel/item")[:max_results]:
        result_url = item.findtext("link", default="")
        results.append(
            {
                "title": item.findtext("title", default=""),
                "url": result_url,
                "snippet": _clean_html(item.findtext("description", default=""))[:500],
                "published_at": "",
                "source": urllib.parse.urlparse(result_url).netloc,
            }
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# NEWS SEARCH — Tiered: NewsAPI → Tavily → Google News RSS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_newsapi_sync(
    query: str,
    max_results: int = 8,
    freshness: str | None = "d",
) -> list[dict[str, Any]]:
    """
    Primary news source: newsapi.org
    Returns articles with image_url field.
    Raises NewsAPIQuotaExceeded on 429 / plan-limit errors.
    """
    from app.config import settings

    api_key = settings.news_api_key.strip()
    if not api_key:
        raise ValueError("NEWS_API_KEY not configured")

    # Map freshness codes → NewsAPI 'from' date offset
    from datetime import datetime, timedelta, timezone
    freshness_days = {"d": 1, "w": 7, "m": 30}.get(freshness or "d", 1)
    from_date = (datetime.now(timezone.utc) - timedelta(days=freshness_days)).strftime("%Y-%m-%d")

    params = urllib.parse.urlencode({
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(max_results, 20),
        "apiKey": api_key,
    })
    url = f"https://newsapi.org/v2/everything?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Omni-Agent/1.0"})

    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (426, 429, 401):
            body = exc.read().decode("utf-8", errors="ignore")
            logger.warning("[newsapi] Quota/auth error %d: %s", exc.code, body[:200])
            raise NewsAPIQuotaExceeded(f"NewsAPI HTTP {exc.code}") from exc
        raise

    status = data.get("status", "")
    if status != "ok":
        code = data.get("code", "")
        if code in ("rateLimited", "maximumResultsReached", "apiKeyExhausted", "apiKeyInvalid", "apiKeyDisabled"):
            raise NewsAPIQuotaExceeded(f"NewsAPI error code: {code}")
        raise RuntimeError(f"NewsAPI returned status={status}, code={code}")

    articles = data.get("articles", [])
    results = []
    for article in articles[:max_results]:
        source_name = (article.get("source") or {}).get("name", "")
        # Skip articles where content is "[Removed]"
        if article.get("title", "") == "[Removed]":
            continue
        results.append({
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "snippet": article.get("description") or article.get("content", ""),
            "published_at": article.get("publishedAt", ""),
            "source": source_name,
            "image_url": article.get("urlToImage") or "",
        })
    logger.info("[newsapi] query='%s' → %d articles", query, len(results))
    return results


def _fetch_tavily_news_sync(
    query: str,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """
    Fallback news source: Tavily search API with topic='news'.
    Returns standardised result dicts (no image_url — Tavily doesn't provide them).
    """
    from app.config import settings

    api_key = settings.tavily_api_key.strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY not configured")

    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "topic": "news",
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Omni-Agent/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_results = data.get("results", [])
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "published_at": r.get("published_date", ""),
            "source": urllib.parse.urlparse(r.get("url", "")).netloc,
            "image_url": "",
        }
        for r in raw_results
    ]
    logger.info("[tavily] query='%s' → %d results", query, len(results))
    return results


def _fetch_ddg_news_sync(
    query: str,
    max_results: int = 8,
    freshness: str | None = "d",
) -> list[dict[str, Any]]:
    """Last-resort news via DuckDuckGo news API."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.news(
                query,
                region="wt-wt",
                safesearch="moderate",
                timelimit=freshness,
                max_results=max_results,
            ))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", ""),
                "published_at": r.get("date", ""),
                "source": r.get("source", ""),
                "image_url": r.get("image", ""),
            }
            for r in raw
        ]
    except Exception as exc:
        logger.warning("[ddg_news] failed: %s", exc)
        return []


def _news_search_sync(
    query: str,
    max_results: int = 8,
    freshness: str | None = "d",
) -> list[dict[str, Any]]:
    """
    Tiered news search:
      1. NewsAPI (primary — has images)
      2. Tavily  (fallback — when quota hit)
      3. DuckDuckGo news (last resort)
      4. Google News RSS (final safety net)
    """
    # 1. Try NewsAPI
    try:
        results = _fetch_newsapi_sync(query, max_results, freshness)
        if results:
            return results
        logger.info("[news] NewsAPI returned 0 results — trying Tavily")
    except NewsAPIQuotaExceeded as exc:
        logger.warning("[news] NewsAPI quota exceeded (%s) — falling back to Tavily", exc)
    except Exception as exc:
        logger.warning("[news] NewsAPI failed (%s) — falling back to Tavily", exc)

    # 2. Try Tavily
    try:
        results = _fetch_tavily_news_sync(query, max_results)
        if results:
            return results
        logger.info("[news] Tavily returned 0 results — trying DuckDuckGo")
    except Exception as exc:
        logger.warning("[news] Tavily failed (%s) — falling back to DuckDuckGo", exc)

    # 3. Try DuckDuckGo news
    results = _fetch_ddg_news_sync(query, max_results, freshness)
    if results:
        return results

    # 4. Google News RSS (final safety net)
    try:
        return _google_news_rss_sync(query, max_results, freshness)
    except Exception as exc:
        logger.error("[news] All news sources failed. Last error: %s", exc)
        return []


async def news_search(
    query: str,
    max_results: int = 8,
    freshness: str | None = "d",
) -> list[dict[str, Any]]:
    """Async news search using the tiered waterfall (NewsAPI → Tavily → DDG → RSS)."""
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, _news_search_sync, query, max_results, freshness
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL WEB SEARCH — DuckDuckGo (unchanged for non-news queries)
# ══════════════════════════════════════════════════════════════════════════════

def _ddg_search_sync(
    query: str,
    max_results: int = 6,
    freshness: str | None = None,
) -> list[dict[str, Any]]:
    """Blocking DuckDuckGo text search for general (non-news) queries."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(
                    query,
                    region="wt-wt",
                    safesearch="moderate",
                    timelimit=freshness,
                    max_results=max_results,
                )
            )
        if not raw_results:
            raise ValueError("DuckDuckGo returned no results")
        return [
            {
                "title": result.get("title", ""),
                "url": result.get("url") or result.get("href", ""),
                "snippet": result.get("body", ""),
                "published_at": result.get("date", ""),
                "source": result.get("source", ""),
                "image_url": "",
            }
            for result in raw_results
        ]
    except Exception as exc:
        logger.warning("[search] DuckDuckGo failed: %s", exc)
        try:
            return _bing_web_rss_sync(query, max_results)
        except Exception as fallback_exc:
            logger.error("[search] Bing RSS fallback failed: %s", fallback_exc)
            return []


async def web_search(
    query: str,
    max_results: int = 6,
    *,
    news: bool = False,
    freshness: str | None = None,
) -> list[dict[str, Any]]:
    """
    General web search via DuckDuckGo (non-news queries).
    For news queries use news_search() instead.
    The 'news' parameter is kept for backward-compatibility but is ignored
    — callers should use news_search() directly for news.
    """
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        _ddg_search_sync,
        query,
        max_results,
        freshness,
    )
    logger.info(
        "[search] query='%s' freshness=%s results=%d",
        query,
        freshness,
        len(results),
    )
    return results


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format general web search results with freshness metadata for LLM grounding."""
    if not results:
        return "No web results found."

    lines = []
    for index, result in enumerate(results, 1):
        lines.append(f"**[{index}] {result.get('title', 'Untitled')}**")
        if result.get("published_at"):
            lines.append(f"Published: {result['published_at']}")
        if result.get("source"):
            lines.append(f"Publisher: {result['source']}")
        lines.append(f"URL: {result.get('url', '')}")
        lines.append(f"{result.get('snippet', '')}\n")
    return "\n".join(lines)


def format_news_results(results: list[dict[str, Any]], retrieved_date: str) -> str:
    """
    Render news cards in a rich format:
      [IMAGE if available]
      • bullet points from snippet
      Source: publisher name  |  📅 date  |  [Read →](url)
      ---
    """
    if not results:
        return "No current news results found."

    lines = [
        "### 📰 Latest News",
        f"*Retrieved: {retrieved_date}*",
        "",
    ]

    for result in results:
        title       = result.get("title", "Untitled")
        pub         = result.get("published_at", "")
        source      = result.get("source", "")
        snippet     = result.get("snippet", "")
        url         = result.get("url", "")
        image_url   = result.get("image_url", "")

        # ── Title ────────────────────────────────────────────────────────
        lines.append(f"#### {title}")

        # ── Image (if available from NewsAPI) ────────────────────────────
        if image_url:
            lines.append(f"![{title}]({image_url})")
            lines.append("")

        # ── Snippet as bullet points ──────────────────────────────────────
        if snippet:
            # Split into sentences and show as bullets (max 3)
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet.strip()) if s.strip()]
            for sent in sentences[:3]:
                lines.append(f"- {sent}")
            lines.append("")

        # ── Meta row: source | date | read link ──────────────────────────
        meta_parts = []
        if source:
            meta_parts.append(f"🗞️ **{source}**")
        if pub:
            # Shorten ISO timestamps: 2024-06-26T12:00:00Z → 2024-06-26
            short_pub = pub[:10] if len(pub) >= 10 else pub
            meta_parts.append(f"📅 {short_pub}")
        if url:
            meta_parts.append(f"[Read article →]({url})")

        if meta_parts:
            lines.append(" · ".join(meta_parts))

        lines.extend(["", "---", ""])

    return "\n".join(lines).rstrip()


# ── Instagram News (targeted DuckDuckGo + Google News) ────────────────────────

def _fetch_instagram_news_sync(
    topic: str = "",
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """
    Fetch Instagram-related news using a two-step approach:
      1. Try Instagram's unofficial public tag page (no auth)
      2. Targeted DuckDuckGo news search
      3. Google News RSS fallback
    Returns a list of result dicts consistent with web_search() format.
    """
    results: list[dict[str, Any]] = []

    # ── Step 1: Try Instagram's unofficial public tag page ─────────────────────
    try:
        import json as _json
        import urllib.request as _ureq

        tag = re.sub(r"\s+", "", topic.lower().strip()) or "trending"
        ig_url = f"https://www.instagram.com/explore/tags/{tag}/?__a=1&__d=dis"
        req = _ureq.Request(
            ig_url,
            headers={
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)",
                "Accept": "application/json",
            },
        )
        with _ureq.urlopen(req, timeout=6) as r:
            data = _json.loads(r.read().decode("utf-8"))

        edges = (
            data.get("graphql", {})
                .get("hashtag", {})
                .get("edge_hashtag_to_media", {})
                .get("edges", [])
        )
        for edge in edges[:5]:
            node = edge.get("node", {})
            caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
            caption = caption_edges[0]["node"]["text"][:300] if caption_edges else ""
            shortcode = node.get("shortcode", "")
            results.append({
                "title": f"Instagram #{tag} post",
                "url": f"https://www.instagram.com/p/{shortcode}/" if shortcode else ig_url,
                "snippet": caption,
                "published_at": "",
                "source": "Instagram",
            })
        if results:
            logger.info("[instagram] Tag page: %d results for '#%s'", len(results), tag)
            return results[:max_results]
    except Exception as exc:
        logger.debug("[instagram] Tag page failed (expected): %s", exc)

    # ── Step 2: DuckDuckGo news search ────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        queries = [
            f"instagram {topic} trending" if topic else "instagram trending viral news today",
            f"instagram {topic} latest" if topic else "instagram latest news today",
        ]
        seen_urls: set[str] = set()
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.news(
                    q,
                    region="wt-wt",
                    safesearch="moderate",
                    timelimit="d",
                    max_results=max_results // 2 + 1,
                ):
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "title": r.get("title", ""),
                            "url": url,
                            "snippet": r.get("body", ""),
                            "published_at": r.get("date", ""),
                            "source": r.get("source", ""),
                        })
        if results:
            logger.info("[instagram] DuckDuckGo: %d results for '%s'", len(results), topic)
            return results[:max_results]
    except Exception as exc:
        logger.warning("[instagram] DuckDuckGo news failed: %s", exc)

    # ── Step 3: Google News RSS fallback ──────────────────────────────────────
    try:
        query = f"instagram {topic}" if topic else "instagram trending"
        results = _google_news_rss_sync(query, max_results, freshness="d")
        logger.info("[instagram] Google News RSS fallback: %d results", len(results))
    except Exception as exc:
        logger.error("[instagram] Google News RSS also failed: %s", exc)

    return results


async def get_instagram_news(topic: str = "", max_results: int = 8) -> list[dict[str, Any]]:
    """Search for Instagram-related news and trending content via DuckDuckGo."""
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, _fetch_instagram_news_sync, topic, max_results
    )
    return results


def format_instagram_results(results: list[dict[str, Any]], topic: str, retrieved_date: str) -> str:
    """Format Instagram news results as rich Markdown."""
    if not results:
        return f"📸 No recent Instagram news found for **{topic or 'trending topics'}**."

    header_topic = f"#{topic}" if topic else "Trending"
    lines = [
        f"### 📸 Instagram News — {header_topic}",
        f"*Retrieved: {retrieved_date} · via DuckDuckGo News*",
        "",
    ]
    for r in results:
        title   = r.get("title", "Post")
        pub     = r.get("published_at", "")
        source  = r.get("source", "")
        snippet = r.get("snippet", "")
        url     = r.get("url", "")

        lines.append(f"#### {title}")
        meta = []
        if pub:
            meta.append(f"📅 {pub}")
        if source:
            meta.append(f"📰 {source}")
        if meta:
            lines.append(" · ".join(meta))
        if snippet:
            lines.append(snippet[:300])
        if url:
            lines.append(f"<small>[Read more →]({url})</small>")
        lines.extend(["", "---", ""])

    return "\n".join(lines).rstrip()
