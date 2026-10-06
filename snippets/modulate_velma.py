"""Velma streaming: transcript, emotion, and named behaviors on a live call.

One WebSocket gives you what was said, how it was said, and whether any of the
behaviors you asked for occurred. Use this during a call. The batch endpoint
takes minutes and is for afterwards.

    velma = Velma(api_key, behaviors=["preset:vishing"], sample_rate=8000,
                  on_event=handle)
    await velma.start()
    velma.feed(pcm)
    await velma.stop()

Events delivered to on_event:
    {"kind": "partial",  "text": ...}
    {"kind": "final",    "text": ..., "speaker": ..., "emotion": ..., "language": ...}
    {"kind": "behavior", "name": ..., "detected": bool, "confidence": float}
    {"kind": "topics",   "topics": [...]}
    {"kind": "summary",  "text": ...}
"""

import asyncio
import json
from typing import Awaitable, Callable

import websockets

URL = "wss://platform.modulate.ai/api/velma-2-streaming"


class Velma:
    def __init__(
        self,
        api_key: str,
        *,
        behaviors: list[str] | None = None,
        sample_rate: int = 8000,
        on_event: Callable[[dict], Awaitable[None] | None] | None = None,
        produce_topics: bool = True,
        produce_summary: bool = True,
    ):
        self._url = (
            f"{URL}?api_key={api_key}"
            f"&audio_format=s16le&sample_rate={sample_rate}&num_channels=1"
        )
        # Velma requires exactly one config text frame before any audio.
        self._config = json.dumps(
            {
                "behaviors": behaviors or [],
                "produce_topics": produce_topics,
                "produce_summary": produce_summary,
            }
        )
        self._on_event = on_event
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._ws = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._ws = await websockets.connect(self._url, max_size=None)
        await self._ws.send(self._config)
        self._tasks = [
            asyncio.create_task(self._send()),
            asyncio.create_task(self._recv()),
        ]

    def feed(self, pcm: bytes) -> None:
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm)
            except Exception:
                pass

    async def _send(self) -> None:
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    await self._ws.send("")
                    return
                await self._ws.send(chunk)
        except Exception:
            pass

    async def _emit(self, event: dict) -> None:
        if not self._on_event:
            return
        result = self._on_event(event)
        if asyncio.iscoroutine(result):
            await result

    async def _recv(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                t = m.get("type")

                if t == "partial_clip":
                    await self._emit({"kind": "partial",
                                      "text": (m.get("partial_clip") or {}).get("text", "")})
                elif t in ("clip", "clip_update"):
                    c = m.get("clip") or m.get("clip_update") or {}
                    await self._emit({
                        "kind": "final" if t == "clip" else "revision",
                        "text": c.get("text", ""),
                        "speaker": c.get("speaker_label"),
                        "emotion": c.get("emotion"),
                        "accent": c.get("accent"),
                        "language": c.get("language"),
                        "deepfake_score": c.get("deepfake_score"),
                    })
                elif t == "behavior_detection":
                    d = m.get("detection") or {}
                    await self._emit({"kind": "behavior",
                                      "name": d.get("behavior_name"),
                                      "detected": bool(d.get("detected")),
                                      "confidence": d.get("confidence"),
                                      "reasoning": d.get("reasoning")})
                elif t == "topics":
                    await self._emit({"kind": "topics", "topics": m.get("topics", [])})
                elif t == "summary":
                    await self._emit({"kind": "summary", "text": m.get("text", "")})
                elif t == "error":
                    await self._emit({"kind": "error", "message": str(m.get("error"))})
        except Exception:
            pass

    async def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        await asyncio.sleep(0.3)
        for t in self._tasks:
            t.cancel()
        if self._ws:
            await self._ws.close()


if __name__ == "__main__":
    import os, sys, wave

    async def main(path):
        async def show(e):
            k = e["kind"]
            if k == "partial":
                print(f"  ... {e['text'][:70]}")
            elif k == "final":
                print(f"  [{e.get('speaker')}] {e['text']}   emotion={e.get('emotion')}")
            elif k == "behavior" and e["detected"]:
                print(f"  >> {e['name']} @ {e['confidence']}")
            elif k in ("topics", "summary", "error"):
                print(f"  {k}: {e.get('topics') or e.get('text') or e.get('message')}")

        with wave.open(path) as w:
            rate, pcm = w.getframerate(), w.readframes(w.getnframes())

        v = Velma(os.environ["MODULATE_API_KEY"],
                  behaviors=["preset:bank_account_holder_impersonation",
                             "preset:vishing", "preset:credential_solicitation"],
                  sample_rate=rate, on_event=show)
        await v.start()
        step = rate // 50 * 2
        for i in range(0, len(pcm), step):
            v.feed(pcm[i:i+step]); await asyncio.sleep(0.002)
        await asyncio.sleep(25)
        await v.stop()

    asyncio.run(main(sys.argv[1] if len(sys.argv)>1 else "samples/synthetic_caller.wav"))
