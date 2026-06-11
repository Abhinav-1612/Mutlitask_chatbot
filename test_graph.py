import asyncio
import logging
from app.graph import get_graph
from app.agents.state import initial_state

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

async def test_query(query: str):
    print("\n" + "="*70)
    print(f"Testing: '{query}'")
    print("="*70)
    
    graph = get_graph()
    state = initial_state(
        session_id="test-1",
        query=query,
    )
    
    try:
        result = await graph.ainvoke(state)
        print("\n✅ SUCCESS!")
        print(f"Route used: {result.get('route_used')}")
        answer = result.get('final_answer', 'NO ANSWER')
        print(f"Answer length: {len(answer)} chars")
        print(f"Answer preview: {answer[:300]}")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_query("What is the weather in London?")
    await test_query("What are the latest news?")
    await test_query("Tell me about Python programming")

if __name__ == "__main__":
    asyncio.run(main())
