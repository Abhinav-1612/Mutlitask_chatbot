from __future__ import annotations

import asyncio
import logging
import urllib.parse
import urllib.request
import json
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

def _fetch_mandi_prices_sync(commodity: str, state: str = "", max_results: int = 15) -> dict[str, Any]:
    """
    Synchronous fetching of Mandi prices from data.gov.in AGMARKNET API.
    """
    api_key = settings.data_gov_api_key
    resource_id = settings.agmarknet_resource_id

    if not api_key or not resource_id:
        return {"error": "API Key or Resource ID not configured. Please add DATA_GOV_API_KEY and AGMARKNET_RESOURCE_ID to your .env file."}

    # Base URL
    url = f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json&limit={max_results}"

    if commodity:
        encoded_commodity = urllib.parse.quote_plus(commodity.title())
        url += f"&filters[commodity]={encoded_commodity}"
    
    if state:
        encoded_state = urllib.parse.quote_plus(state.title())
        url += f"&filters[state]={encoded_state}"

    import time
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                
            if data.get("status") == "ok":
                return {"status": "ok", "records": data.get("records", [])}
            else:
                return {"error": data.get("message", "API request failed")}
                
        except urllib.error.HTTPError as e:
            if e.code == 502 and attempt < 2:
                logger.warning("[agriculture] 502 Bad Gateway from AGMARKNET, retrying...")
                time.sleep(1.5)
                continue
            logger.error("[agriculture] API fetch failed: %s", e)
            return {"error": f"Failed to fetch data: HTTP {e.code}"}
        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5)
                continue
            logger.error("[agriculture] API fetch failed: %s", exc)
            return {"error": f"Failed to fetch data: {str(exc)}"}
    
    return {"error": "Failed to fetch data after retries"}

async def get_mandi_prices(commodity: str, state: str = "", max_results: int = 15) -> dict[str, Any]:
    """
    Async wrapper to fetch mandi prices.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_mandi_prices_sync, commodity, state, max_results)

def format_mandi_prices(data: dict[str, Any], commodity: str) -> str:
    """
    Format the AGMARKNET API response into a readable summary.
    """
    if "error" in data:
        return f"Could not retrieve mandi prices: {data['error']}"

    records = data.get("records", [])
    if not records:
        return f"No recent mandi prices found for {commodity}."

    lines = [
        f"## 🌾 Latest Mandi Prices for {commodity.title()}",
        "",
        "| State | District | Market (Mandi) | Arrival Date | Min Price (₹/Q) | Max Price (₹/Q) | Modal Price (₹/Q) |",
        "|-------|----------|----------------|--------------|-----------------|-----------------|-------------------|"
    ]

    for rec in records:
        state = rec.get("state", "Unknown")
        district = rec.get("district", "Unknown")
        market = rec.get("market", "Unknown")
        date = rec.get("arrival_date", "Unknown")
        min_p = rec.get("min_price", "-")
        max_p = rec.get("max_price", "-")
        mod_p = rec.get("modal_price", "-")
        
        lines.append(f"| {state} | {district} | {market} | {date} | {min_p} | {max_p} | {mod_p} |")

    lines.extend(["", "*Source: AGMARKNET via data.gov.in*"])
    return "\n".join(lines)
