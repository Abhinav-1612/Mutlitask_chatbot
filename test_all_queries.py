import asyncio
import json
import requests
import time
from app.sse import iter_sse_events

def test_all_queries():
    """Test weather, news, and general questions."""
    time.sleep(2)
    
    test_cases = [
        ("What is the weather in London?", "weather"),
        ("Tell me the latest news", "news"),
        ("Explain quantum computing", "general"),
    ]
    
    for query, query_type in test_cases:
        print("\n" + "="*70)
        print(f"Test: {query_type.upper()}")
        print(f"Query: {query}")
        print("="*70)
        
        try:
            params = {
                "session_id": f"test-{query_type}",
                "message": query,
                "file_ids": "",
                "active_url": "",
            }
            
            response = requests.get(
                "http://localhost:8000/chat/stream",
                params=params,
                stream=True,
                timeout=120,
            )
            
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                continue
            
            event_count = 0
            final_result = None
            
            for event in iter_sse_events(response.iter_lines(decode_unicode=True)):
                event_count += 1
                if event["event"] == "result":
                    final_result = json.loads(event["data"])
            
            if final_result and final_result.get("answer"):
                route = final_result.get("route", "unknown")
                answer_len = len(final_result.get("answer", ""))
                print(f"✅ SUCCESS!")
                print(f"   Route: {route}")
                print(f"   Answer length: {answer_len} chars")
                print(f"   Events received: {event_count}")
                print(f"   Preview: {final_result.get('answer', '')[:150]}")
            else:
                print(f"❌ No answer received (Events: {event_count})")
                
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {str(e)[:100]}")

if __name__ == "__main__":
    print("🚀 Testing all query types...\n")
    test_all_queries()
    print("\n✅ All tests completed!")
