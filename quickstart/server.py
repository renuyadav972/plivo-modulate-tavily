"""Get a real phone call into your agent in under ten minutes.

This is deliberately one small file with no AI dependencies. Run it, point a
Plivo number at it, call the number, and you will hear your own voice played
back. That proves the whole round trip works before you plug in anything
clever.

    python quickstart/server.py
    ngrok http 7860
    # set the ngrok host in .env as PLIVO_WEBHOOK_HOST, then restart
    # point your Plivo number's Answer URL at https://<host>/answer

Then replace EchoAgent with your own stack. You only have to fill in two
methods, and the protocol details stay handled for you.
"""

import argparse
import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger

import plivo_audio as plivo

load_dotenv(override=True)

app = FastAPI()

# Plivo streams 8kHz mu-law. Each inbound frame is typically 20ms of audio.
SAMPLE_RATE = 8000


# --- plug your agent in here ------------------------------------------------


class VoiceAgent:
    """Implement these two methods and you have a phone agent.

    on_audio is called roughly every 20ms with 16-bit PCM at 8kHz.
    Call send(pcm) whenever you want the caller to hear something.
    Call interrupt() to drop audio you already queued, which is barge-in.
    """

    def __init__(self, send, interrupt):
        self.send = send
        self.interrupt = interrupt

    async def on_start(self, call_id: str) -> None:
        """The call has connected."""

    async def on_audio(self, pcm: bytes) -> None:
        """A chunk of the caller's speech, 16-bit PCM at 8kHz."""

    async def on_dtmf(self, digit: str) -> None:
        """The caller pressed a key."""

    async def on_end(self) -> None:
        """The call is over. Flush anything you were holding."""


class EchoAgent(VoiceAgent):
    """Records until you stop talking, then plays it back.

    Not useful on its own. It exists to prove, with zero API keys, that audio is
    flowing in both directions before you start debugging a model.
    """

    SILENCE_RMS = 500          # below this counts as silence
    SILENCE_FRAMES = 25        # ~0.5s of silence ends a turn
    MIN_SPEECH_FRAMES = 10     # ignore blips shorter than ~0.2s

    def __init__(self, send, interrupt):
        super().__init__(send, interrupt)
        self._buffer = bytearray()
        self._quiet = 0
        self._speech = 0

    async def on_start(self, call_id: str) -> None:
        logger.info(f"Echo agent ready on call {call_id}. Say something.")

    async def on_audio(self, pcm: bytes) -> None:
        loud = plivo.rms(pcm) > self.SILENCE_RMS

        if loud:
            if self._speech == 0:
                # The caller started talking, so stop whatever we were playing.
                await self.interrupt()
            self._buffer.extend(pcm)
            self._speech += 1
            self._quiet = 0
            return

        if self._speech == 0:
            return  # still waiting for the caller to begin

        self._buffer.extend(pcm)
        self._quiet += 1

        if self._quiet >= self.SILENCE_FRAMES:
            if self._speech >= self.MIN_SPEECH_FRAMES:
                logger.info(f"Echoing back {len(self._buffer) / 2 / SAMPLE_RATE:.1f}s")
                await self.send(bytes(self._buffer))
            self._buffer.clear()
            self._speech = 0
            self._quiet = 0


# Swap this for your own class once the echo works.
AGENT = EchoAgent


# --- Plivo plumbing, you should not need to touch this ----------------------


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request):
    """Plivo fetches this when a call arrives, and we tell it where to stream."""
    host = os.getenv("PLIVO_WEBHOOK_HOST", "")
    if not host:
        logger.error("PLIVO_WEBHOOK_HOST is not set in .env")
        return Response(
            content="<Response><Speak>Server is not configured.</Speak></Response>",
            media_type="application/xml",
        )

    url = f"wss://{host}/stream"
    logger.info(f"Call incoming, streaming audio to {url}")
    return Response(content=plivo.answer_xml(url), media_type="application/xml")


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Plivo connected")

    stream_id: str | None = None

    async def send(pcm: bytes) -> None:
        """Play PCM to the caller, in chunks Plivo is happy to buffer."""
        if stream_id is None:
            return
        # 8000 samples/sec * 2 bytes = 16000 bytes/sec. 3200 bytes is 200ms.
        for i in range(0, len(pcm), 3200):
            await websocket.send_text(plivo.play_audio(stream_id, pcm[i : i + 3200]))

    async def interrupt() -> None:
        if stream_id is not None:
            await websocket.send_text(plivo.clear_audio(stream_id))

    agent = AGENT(send, interrupt)

    try:
        while True:
            message = await websocket.receive_text()
            event = plivo.parse(message)

            if event["type"] == "start":
                stream_id = event["stream_id"]
                logger.info(f"Stream {stream_id} on call {event['call_id']}")
                await agent.on_start(event["call_id"])

            elif event["type"] == "audio":
                await agent.on_audio(event["pcm"])

            elif event["type"] == "dtmf":
                logger.info(f"DTMF {event['digit']}")
                await agent.on_dtmf(event["digit"])

    except WebSocketDisconnect:
        logger.info("Caller hung up")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        await agent.on_end()


@app.get("/health")
async def health():
    return {
        "plivo_configured": bool(os.getenv("PLIVO_WEBHOOK_HOST")),
        "agent": AGENT.__name__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plivo audio streaming quickstart")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logger.remove(0)
    logger.add(sys.stderr, level="INFO")

    host = os.getenv("PLIVO_WEBHOOK_HOST")
    if host:
        logger.info(f"Point your Plivo number's Answer URL at https://{host}/answer")
    else:
        logger.warning("PLIVO_WEBHOOK_HOST is not set. Start a tunnel and set it in .env")

    uvicorn.run(app, host=args.host, port=args.port)
