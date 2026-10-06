"""HTTP + WebSocket server for the demo: Plivo call legs and the stage dashboard."""

import argparse
import asyncio
import os
import ssl
import sys

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from loguru import logger

import events

# The voice pipeline pulls in heavy dependencies. Import it defensively so the
# dashboard and rehearsal mode still run on a machine where pipecat is not
# installed yet, which is what you want while building slides.
try:
    from bot import run_phone_bot
except Exception as _bot_import_error:  # pragma: no cover
    run_phone_bot = None
    logger.warning(f"Voice pipeline unavailable: {_bot_import_error}")

load_dotenv(override=True)

app = FastAPI()


@app.get("/")
async def dashboard():
    """The stage view. Open this on the projector before dialling in."""
    return FileResponse("dashboard/index.html")


@app.get("/health")
async def health():
    """Quick pre-flight check that every key is present before you go on stage."""
    return {
        "plivo": bool(os.getenv("PLIVO_AUTH_ID") and os.getenv("PLIVO_AUTH_TOKEN")),
        "modulate": bool(os.getenv("MODULATE_API_KEY")),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "deepgram": bool(os.getenv("DEEPGRAM_API_KEY")),
        "cartesia": bool(os.getenv("CARTESIA_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "webhook_host": os.getenv("PLIVO_WEBHOOK_HOST") or None,
    }


# --- Plivo call legs ----------------------------------------------------


@app.api_route("/answer", methods=["GET", "POST"])
@app.post("/plivo-webhook")
async def plivo_webhook(request: Request):
    """Tell Plivo to open a bidirectional audio stream to our /ws endpoint."""
    host = os.getenv("PLIVO_WEBHOOK_HOST", "")
    if not host:
        logger.error("PLIVO_WEBHOOK_HOST is not set")
        return Response(
            content="<Response><Speak>The demo server is not configured.</Speak></Response>",
            media_type="application/xml",
        )

    logger.info(f"Plivo webhook hit, streaming audio to wss://{host}/ws")

    xml = (
        "<Response>"
        '<Stream bidirectional="true" keepCallAlive="true" '
        'contentType="audio/x-mulaw;rate=8000" streamTimeout="86400">'
        f"wss://{host}/ws"
        "</Stream>"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def plivo_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Plivo WebSocket accepted")
    if run_phone_bot is None:
        logger.error("Voice pipeline not installed, run pip install -r requirements.txt")
        await websocket.close()
        return
    try:
        await run_phone_bot(websocket)
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        logger.info("Plivo WebSocket closed")


@app.post("/rehearse")
async def rehearse():
    """Replay a scripted call onto the dashboard, with no phone and no API spend.

    Use this to rehearse the stage sequence and to check the projector before
    the room fills up.
    """
    import simulate

    asyncio.create_task(simulate.run())
    return {"status": "rehearsing"}


@app.post("/call")
async def place_call(request: Request):
    """Place an outbound call so the demo can dial the audience instead."""
    import plivo

    body = await request.json()
    to_number = (body or {}).get("to", "").strip()
    if not to_number:
        return Response(
            content='{"error": "missing to number"}',
            media_type="application/json",
            status_code=400,
        )

    host = os.getenv("PLIVO_WEBHOOK_HOST", "")
    client = plivo.RestClient(
        os.getenv("PLIVO_AUTH_ID"), os.getenv("PLIVO_AUTH_TOKEN")
    )
    resp = client.calls.create(
        from_=os.getenv("PLIVO_PHONE_NUMBER", ""),
        to_=to_number,
        answer_url=f"https://{host}/plivo-webhook",
        answer_method="POST",
    )
    logger.info(f"Outbound call to {to_number}: {resp}")
    return {"status": "calling", "to": to_number}


# --- Dashboard feed -----------------------------------------------------


@app.websocket("/events")
async def events_websocket(websocket: WebSocket):
    """Stream live call events to the dashboard."""
    await websocket.accept()
    queue = events.subscribe()

    try:
        # Replay so a browser refresh mid-demo does not land on a blank screen.
        for event in events.history():
            await websocket.send_json(event)

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                await websocket.send_json({"lane": "ping", "kind": "ping", "data": {}})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Dashboard socket closed: {e}")
    finally:
        events.unsubscribe(queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plivo + Modulate + Tavily demo")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    logger.add(sys.stderr, level="TRACE" if args.verbose else "DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)
