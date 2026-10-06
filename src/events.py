"""In-process event bus so the stage dashboard can watch the live call.

The bot publishes; the /events WebSocket subscribes. Nothing here is required
for the voice agent to work, it exists so an audience can see the three layers
light up in real time.
"""

import asyncio
import time
from typing import Any

from loguru import logger

_subscribers: set[asyncio.Queue] = set()

# Replayed to any dashboard that connects mid-call, so a late browser refresh
# during the demo does not show an empty screen.
_history: list[dict[str, Any]] = []
_HISTORY_LIMIT = 200


def subscribe() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    _subscribers.discard(queue)


def history() -> list[dict[str, Any]]:
    return list(_history)


def reset() -> None:
    """Clear history at the start of a call so each demo run is clean."""
    _history.clear()
    publish("call", "start", {"message": "Call connected"})


def publish(lane: str, kind: str, data: dict[str, Any] | None = None) -> None:
    """Fan an event out to every connected dashboard.

    lane is one of: call, plivo, modulate, tavily, agent, guardrail.
    """
    event = {
        "lane": lane,
        "kind": kind,
        "ts": time.time(),
        "data": data or {},
    }

    _history.append(event)
    if len(_history) > _HISTORY_LIMIT:
        del _history[0 : len(_history) - _HISTORY_LIMIT]

    dead = []
    for queue in _subscribers:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # A stalled browser tab must never back-pressure the audio path.
            dead.append(queue)
    for queue in dead:
        logger.warning("Dropping a stalled dashboard subscriber")
        _subscribers.discard(queue)
