"""Grounded answers for a voice agent, with the latency settings that work.

The naive settings are too slow for a phone call. Measured against the live API
on 2026-10-05, same query, three runs each:

    search_depth   include_answer   median   worst
    advanced       advanced          5.46s   5.79s   caller thinks you hung up
    basic          basic             3.80s   4.60s   still too slow
    fast           advanced          1.27s   1.53s   <- use this
    fast           basic             0.47s   1.75s   fastest, thinner answer

"fast" is a different index, not a truncated "advanced", so answers hold up.

    answer = await search("what is plivo audio streaming")
    answer = await search("how do I start a stream", only=["plivo.com"])
"""

import asyncio
import os

from tavily import TavilyClient

_client: TavilyClient | None = None


def _get() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


async def search(
    query: str,
    *,
    only: list[str] | None = None,
    topic: str = "general",
    max_results: int = 5,
) -> dict:
    """Search and return {"answer": str, "sources": [{"title","url"}], "seconds": float}.

    only restricts to specific domains, which is how you build a docs-grounded
    agent that can cite the page it used.
    """
    kwargs = {
        "query": query,
        "search_depth": "fast",
        "include_answer": "advanced",
        "max_results": max_results,
        "topic": topic,
    }
    if only:
        kwargs["include_domains"] = only
        kwargs["include_domains_mode"] = "restrict"

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    # The SDK is synchronous. Keep it off the loop that is carrying your audio.
    response = await asyncio.to_thread(_get().search, **kwargs)

    return {
        "answer": (response.get("answer") or "").strip(),
        "sources": [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in (response.get("results") or [])[:4]
        ],
        "seconds": round(loop.time() - t0, 2),
    }


def to_speech(result: dict) -> str:
    """Flatten into something an LLM can read out loud and attribute."""
    if not result["answer"]:
        return "I could not find a reliable source for that."
    names = ", ".join(s["title"] for s in result["sources"][:2] if s["title"])
    return result["answer"] + (f"\nSources: {names}." if names else "")


# --- the tool definition, for function-calling agents ----------------------

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_the_web",
        "description": (
            "Search the live web for current facts, pricing, product details, or "
            "anything you are not certain of. Use this instead of guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused query, not the caller's whole sentence.",
                }
            },
            "required": ["query"],
        },
    },
}


if __name__ == "__main__":
    async def main():
        for q, only in [("what is plivo audio streaming", None),
                        ("how do I start an audio stream on a live call", ["plivo.com"])]:
            r = await search(q, only=only)
            print(f"\n[{r['seconds']}s] {q}")
            print(" ", r["answer"][:180])
            for s in r["sources"][:2]:
                print("   -", s["url"][:70])
    asyncio.run(main())
