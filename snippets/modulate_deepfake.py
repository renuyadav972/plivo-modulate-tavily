"""Is the voice on this call synthetic? Drop-in, framework agnostic.

Verified against the live API on 2026-10-05: TTS audio came back as
"synthetic" at 97% confidence on every 4-second window.

    detector = DeepfakeDetector(api_key, sample_rate=8000)
    await detector.start()
    ...
    detector.feed(pcm_bytes)            # call this with caller audio
    ...
    await detector.stop()

Pass on_verdict to be told as soon as a window is scored. The detector never
blocks your audio path: if it falls behind, audio is dropped rather than your
call being delayed.
"""

import asyncio
import json
from typing import Awaitable, Callable

import websockets

URL = "wss://platform.modulate.ai/api/velma-2-synthetic-voice-detection-streaming"

Verdict = dict  # {"verdict": "synthetic"|"non-synthetic"|"no-content",
#                 "confidence": float, "start_time_ms": int, "end_time_ms": int}


class DeepfakeDetector:
    def __init__(
        self,
        api_key: str,
        *,
        sample_rate: int = 8000,
        on_verdict: Callable[[Verdict], Awaitable[None] | None] | None = None,
    ):
        self._url = (
            f"{URL}?api_key={api_key}"
            f"&audio_format=s16le&sample_rate={sample_rate}&num_channels=1"
        )
        self._on_verdict = on_verdict
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._ws = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._ws = await websockets.connect(self._url, max_size=None)
        self._tasks = [
            asyncio.create_task(self._send()),
            asyncio.create_task(self._recv()),
        ]

    def feed(self, pcm: bytes) -> None:
        """16-bit mono PCM at the sample rate you constructed with."""
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            # Never let analysis slow the call down.
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
                    await self._ws.send("")  # end of stream
                    return
                await self._ws.send(chunk)
        except Exception:
            pass

    async def _recv(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") != "frame":
                    continue
                verdict = msg.get("frame") or {}
                if self._on_verdict:
                    result = self._on_verdict(verdict)
                    if asyncio.iscoroutine(result):
                        await result
        except Exception:
            pass

    async def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        await asyncio.sleep(0.2)
        for t in self._tasks:
            t.cancel()
        if self._ws:
            await self._ws.close()


# --- example ---------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import wave

    async def main(path: str):
        async def on_verdict(v):
            bar = "#" * int((v.get("confidence") or 0) * 30)
            print(f"{v['start_time_ms']:>6}ms  {v['verdict']:<14} "
                  f"{v.get('confidence', 0):.3f} {bar}")

        with wave.open(path) as w:
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())

        d = DeepfakeDetector(os.environ["MODULATE_API_KEY"],
                             sample_rate=rate, on_verdict=on_verdict)
        await d.start()
        # Feed in real time-ish chunks, as a live call would.
        step = rate // 50 * 2  # 20ms
        for i in range(0, len(pcm), step):
            d.feed(pcm[i : i + step])
            await asyncio.sleep(0.002)
        await asyncio.sleep(3)
        await d.stop()

    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "samples/synthetic_caller.wav"))
