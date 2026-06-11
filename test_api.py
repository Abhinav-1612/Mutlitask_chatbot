import requests
import time
import json
from app.sse import iter_sse_events

def test_sse_endpoint():
    print("Waiting for server to warm up...")
    time.sleep(3)
    
    print("\n" + "="*70)
    print("Testing: /chat/stream endpoint with weather query")
    print("="*70)
    
    try:
        params = {
            "session_id": "test-session",
            "message": "What is the weather in London?",
            "file_ids": "",
            "active_url": "",
        }
        
        response = requests.get(
            "http://localhost:8000/chat/stream",
            params=params,
            stream=True,
            timeout=60,
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}\n")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return
        
        event_count = 0
        final_result = None
        
        for event in iter_sse_events(response.iter_lines(decode_unicode=True)):
            event_count += 1
            event_type = event["event"]
            event_data = event["data"]
            
            print(f"\n[Event #{event_count}] Type: {event_type}")
            
            if event_type == "log":
                data = json.loads(event_data)
                print(f"  Node: {data.get('node')}")
                print(f"  Message: {data.get('message')[:100]}")
            elif event_type == "result":
                final_result = json.loads(event_data)
                print(f"  Route: {final_result.get('route')}")
                print(f"  Answer length: {len(final_result.get('answer', ''))}")
                print(f"  Answer preview: {final_result.get('answer', '')[:200]}")
                print(f"  Sources count: {len(final_result.get('sources', []))}")
            elif event_type == "error":
                print(f"  Error: {event_data}")
        
        print(f"\n✅ Success! Received {event_count} events")
        if final_result:
            print(f"✅ Final answer received: {len(final_result.get('answer', ''))} chars")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sse_endpoint()
