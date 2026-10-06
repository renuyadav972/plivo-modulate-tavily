"""Plivo Audio Streaming, with no dependencies beyond a web framework.

Plivo speaks a small JSON protocol over the WebSocket it opens into your server.
This module is the whole protocol, written out so you can read it in one sitting
and port it to whatever stack you brought tonight.

What Plivo sends you:

    {"event": "start", "start": {"streamId": "...", "callId": "..."}}
    {"event": "media", "media": {"payload": "<base64 mu-law 8kHz>"}}
    {"event": "dtmf",  "dtmf":  {"digit": "1"}}

What you send back:

    {"event": "playAudio",  "streamId": "...",
     "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000,
               "payload": "<base64 mu-law 8kHz>"}}
    {"event": "clearAudio", "streamId": "..."}

clearAudio is how you barge in: it drops whatever you already queued so the
caller can interrupt your agent. Send it the moment you detect speech.

Audio on the wire is 8kHz mu-law, which is what the phone network itself uses.
The helpers below convert to and from 16-bit PCM, which is what almost every
speech API wants.
"""

import base64
import json
from typing import Any

# --- mu-law conversion ------------------------------------------------------
# Built once at import. No numpy, no audioop (which is gone in Python 3.13).

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def _build_decode_table() -> list[int]:
    table = []
    for byte in range(256):
        value = ~byte & 0xFF
        sign = value & 0x80
        exponent = (value >> 4) & 0x07
        mantissa = value & 0x0F
        sample = ((mantissa << 3) + _MULAW_BIAS) << exponent
        sample -= _MULAW_BIAS
        table.append(-sample if sign else sample)
    return table


_DECODE_TABLE = _build_decode_table()

# Standard G.711 exponent lookup, indexed by (sample >> 7) & 0xFF.
# Segment sizes double each step and must total 256 entries.
_EXPONENT_TABLE = (
    [0] * 2 + [1] * 2 + [2] * 4 + [3] * 8
    + [4] * 16 + [5] * 32 + [6] * 64 + [7] * 128
)
assert len(_EXPONENT_TABLE) == 256


def mulaw_to_pcm(payload: bytes) -> bytes:
    """8kHz mu-law bytes to 16-bit little-endian PCM."""
    out = bytearray(len(payload) * 2)
    for i, byte in enumerate(payload):
        sample = _DECODE_TABLE[byte]
        out[2 * i] = sample & 0xFF
        out[2 * i + 1] = (sample >> 8) & 0xFF
    return bytes(out)


def pcm_to_mulaw(pcm: bytes) -> bytes:
    """16-bit little-endian PCM to 8kHz mu-law bytes."""
    out = bytearray(len(pcm) // 2)
    for i in range(len(out)):
        sample = int.from_bytes(pcm[2 * i : 2 * i + 2], "little", signed=True)

        sign = 0x80 if sample < 0 else 0x00
        if sample < 0:
            sample = -sample
        if sample > _MULAW_CLIP:
            sample = _MULAW_CLIP
        sample += _MULAW_BIAS

        exponent = _EXPONENT_TABLE[(sample >> 7) & 0xFF]
        mantissa = (sample >> (exponent + 3)) & 0x0F
        out[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return bytes(out)


def rms(pcm: bytes) -> float:
    """Rough loudness of a PCM chunk, for simple silence detection."""
    if len(pcm) < 2:
        return 0.0
    total = 0
    count = len(pcm) // 2
    for i in range(count):
        sample = int.from_bytes(pcm[2 * i : 2 * i + 2], "little", signed=True)
        total += sample * sample
    return (total / count) ** 0.5


# --- protocol ---------------------------------------------------------------


def answer_xml(websocket_url: str) -> str:
    """The XML Plivo fetches when a call comes in.

    bidirectional=true is what lets you send audio back. Without it you can
    listen but your agent cannot speak.
    """
    return (
        "<Response>"
        f'<Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000" streamTimeout="86400">'
        f"{websocket_url}"
        f"</Stream>"
        "</Response>"
    )


def parse(message: str) -> dict[str, Any]:
    """Normalise one inbound Plivo frame.

    Returns a dict with "type" of start, audio, dtmf, or other.
    Audio frames carry "pcm", already converted for you.
    """
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return {"type": "other"}

    event = data.get("event")

    if event == "media":
        payload = (data.get("media") or {}).get("payload")
        if not payload:
            return {"type": "other"}
        mulaw = base64.b64decode(payload)
        return {"type": "audio", "mulaw": mulaw, "pcm": mulaw_to_pcm(mulaw)}

    if event == "start":
        start = data.get("start") or {}
        return {
            "type": "start",
            "stream_id": start.get("streamId"),
            "call_id": start.get("callId"),
        }

    if event == "dtmf":
        return {"type": "dtmf", "digit": (data.get("dtmf") or {}).get("digit")}

    return {"type": "other", "event": event}


def play_audio(stream_id: str, pcm: bytes) -> str:
    """Wrap PCM for sending back to the caller."""
    return json.dumps(
        {
            "event": "playAudio",
            "streamId": stream_id,
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": base64.b64encode(pcm_to_mulaw(pcm)).decode(),
            },
        }
    )


def clear_audio(stream_id: str) -> str:
    """Drop queued audio so the caller can interrupt. This is barge-in."""
    return json.dumps({"event": "clearAudio", "streamId": stream_id})
