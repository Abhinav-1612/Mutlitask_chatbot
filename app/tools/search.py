"""Fresh web/news search and structured current-weather tools."""
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


def _ddg_search_sync(
    query: str,
    max_results: int = 6,
    news: bool = False,
    freshness: str | None = None,
) -> list[dict[str, Any]]:
    """Blocking DuckDuckGo search with a news-specific freshness path."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            if news:
                raw_results = list(
                    ddgs.news(
                        query,
                        region="wt-wt",
                        safesearch="moderate",
                        timelimit=freshness,
                        max_results=max_results,
                    )
                )
            else:
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
            }
            for result in raw_results
        ]
    except Exception as exc:
        logger.warning("[search] DuckDuckGo failed: %s", exc)
        try:
            if news:
                return _google_news_rss_sync(query, max_results, freshness)
            return _bing_web_rss_sync(query, max_results)
        except Exception as fallback_exc:
            logger.error("[search] Search fallback failed: %s", fallback_exc)
            return []


async def web_search(
    query: str,
    max_results: int = 6,
    *,
    news: bool = False,
    freshness: str | None = None,
) -> list[dict[str, Any]]:
    """Search the web, optionally using a freshness-filtered news index."""
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        _ddg_search_sync,
        query,
        max_results,
        news,
        freshness,
    )
    logger.info(
        "[search] query='%s' news=%s freshness=%s results=%d",
        query,
        news,
        freshness,
        len(results),
    )
    return results


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results with freshness metadata for LLM grounding."""
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
    """Render fresh news directly so dates and source links are never lost."""
    if not results:
        return "No current news results found."

    lines = [
        f"### 📰 Latest News",
        f"*Retrieved: {retrieved_date}*",
        "",
    ]
    for result in results:
        title = result.get('title', 'Untitled')
        pub = result.get('published_at', '')
        source = result.get('source', '')
        snippet = result.get('snippet', '')
        url = result.get('url', '')

        lines.append(f"#### {title}")
        meta_parts = []
        if pub:
            meta_parts.append(f"📅 {pub}")
        if source:
            meta_parts.append(f"🗞️ {source}")
        if meta_parts:
            lines.append(" · ".join(meta_parts))
        if snippet:
            lines.append(snippet)
        if url:
            lines.append(f"<small>[Read source →]({url})</small>")
        lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip()
