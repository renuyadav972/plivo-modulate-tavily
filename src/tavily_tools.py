"""Tavily as the agent's live knowledge, exposed to the LLM as function calls.

Two tools, deliberately kept distinct:

  search_the_web   broad, current, cited answers from the open web
  search_the_docs  the same engine restricted to Plivo's own documentation

The second one matters more than it looks. Restricting the index is what turns
"a chatbot that sounds confident" into "a support agent that cites a page you
can open", which is the whole point of the Tavily layer on stage.

API shape follows https://docs.tavily.com/documentation/api-reference
"""

import asyncio
import os

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from tavily import TavilyClient

import events

_client: TavilyClient | None = None

DOC_DOMAINS = ["plivo.com", "docs.plivo.com"]


def _get_client() -> TavilyClient | None:
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set, web grounding disabled")
            return None
        _client = TavilyClient(api_key=api_key)
    return _client


def _format(response: dict) -> str:
    """Flatten a Tavily response into something speakable and grounded.

    The spoken answer comes from Tavily's own synthesis, and the source titles
    ride along so the model can attribute out loud rather than assert.
    """
    answer = (response.get("answer") or "").strip()
    results = response.get("results") or []

    sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "score": r.get("score"),
        }
        for r in results[:4]
    ]

    events.publish(
        "tavily",
        "results",
        {
            "query": response.get("query", ""),
            "answer": answer,
            "sources": sources,
            "response_time": response.get("response_time"),
        },
    )

    if not answer and not sources:
        return "No reliable sources were found for that question."

    lines = []
    if answer:
        lines.append(answer)
    if sources:
        names = ", ".join(s["title"] for s in sources[:2] if s["title"])
        if names:
            lines.append(f"Sources: {names}.")
    return "\n".join(lines)


async def _search(query: str, *, restrict_to_docs: bool, topic: str = "general") -> str:
    client = _get_client()
    if client is None:
        return (
            "Web search is not configured right now, so answer from what you "
            "already know and say you are not certain."
        )

    events.publish(
        "tavily",
        "query",
        {"query": query, "scope": "docs" if restrict_to_docs else "web"},
    )

    # Latency matters more than depth here, because the caller is sitting in
    # silence while this runs. Measured against the live API on 2026-10-05:
    #
    #   search_depth   include_answer   median   worst
    #   advanced       advanced          5.46s   5.79s   unusable on a call
    #   basic          basic             3.80s   4.60s   unusable on a call
    #   fast           advanced          1.27s   1.53s   <- what we use
    #   fast           basic             0.47s   1.75s   fastest, thinner answer
    #
    # "fast" is a different index, not a truncated "advanced", so the answer is
    # still good. Drop to include_answer="basic" if you need another 800ms.
    kwargs = {
        "query": query,
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "fast"),
        "include_answer": "advanced",
        "max_results": 5,
        "topic": topic,
    }
    if restrict_to_docs:
        kwargs["include_domains"] = DOC_DOMAINS
        kwargs["include_domains_mode"] = "restrict"

    try:
        # The SDK is synchronous, so keep it off the event loop that is also
        # carrying live audio.
        response = await asyncio.to_thread(client.search, **kwargs)
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        events.publish("tavily", "error", {"message": str(e)})
        return "The web search failed. Tell the caller you could not look that up."

    return _format(response)


# --- Pipecat function handlers ------------------------------------------


async def search_the_web(params):
    query = params.arguments.get("query", "")
    topic = params.arguments.get("topic", "general")
    result = await _search(query, restrict_to_docs=False, topic=topic)
    await params.result_callback({"result": result})


async def search_the_docs(params):
    query = params.arguments.get("query", "")
    result = await _search(query, restrict_to_docs=True)
    await params.result_callback({"result": result})


web_schema = FunctionSchema(
    name="search_the_web",
    description=(
        "Search the live web for current facts, news, pricing, product "
        "announcements, or anything that may have changed recently. Use this "
        "whenever the caller asks about something you are not certain of, "
        "rather than guessing."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "A focused search query, not the caller's full sentence.",
        },
        "topic": {
            "type": "string",
            "enum": ["general", "news"],
            "description": "Use 'news' for recent events and announcements.",
        },
    },
    required=["query"],
)

docs_schema = FunctionSchema(
    name="search_the_docs",
    description=(
        "Search Plivo's own documentation and website for how a Plivo product, "
        "API, or feature works. Prefer this over search_the_web for any "
        "question about Plivo itself, so the answer is citable."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "A focused search query about a Plivo product or API.",
        }
    },
    required=["query"],
)

TOOLS = ToolsSchema(standard_tools=[web_schema, docs_schema])
