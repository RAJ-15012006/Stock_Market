from phi.tools.duckduckgo import DuckDuckGo
import json

def test_search():
    ddg = DuckDuckGo()
    try:
        results = ddg.duckduckgo_search("latest crypto news")
        print(f"Search successful: {len(results)} results")
        print(results[:2])
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_search()
