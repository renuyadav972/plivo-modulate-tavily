# Ship a voice agent that answers a real phone call

Built for **Ship an AI Voice Agent Workshop**, SF Tech Week, 6 October 2026.

You brought your own voice agent. This gets it onto a real phone number, and
adds the guardrails tonight is judged on: fraud detection, hallucination
prevention, and compliance.

| You want | Go to | Time |
|---|---|---|
| My agent answering a real phone call | [`quickstart/`](quickstart/) | ~10 min |
| Guardrails on top of my existing stack | [`snippets/`](snippets/) | ~15 min |
| A finished agent to read or fork | [`reference/`](reference/) | clone and run |

---

## Quickstart: phone call in, audio out

No AI dependencies, no API keys. Prove the round trip first, then plug your
agent in. The whole protocol is in one readable file,
[`quickstart/plivo_audio.py`](quickstart/plivo_audio.py).

```bash
git clone <this repo> && cd trusted-voice-agent
uv venv --python 3.12 && uv pip install -r requirements.txt
cp .env.example .env
```

Start a tunnel, because Plivo has to reach your laptop:

```bash
ngrok http 7860
```

Put the ngrok host in `.env` as `PLIVO_WEBHOOK_HOST` (no `https://`), then:

```bash
python quickstart/server.py
```

Point your Plivo number's Answer URL at `https://<your-host>/answer`, call it,
and you will hear your own voice echoed back. That means audio is flowing both
ways and you can start building.

### Plugging in your own agent

Replace `EchoAgent` in `quickstart/server.py`. You implement two methods:

```python
class MyAgent(VoiceAgent):
    async def on_audio(self, pcm: bytes) -> None:
        # 20ms of the caller, 16-bit PCM at 8kHz. Feed it to your STT.
        ...

    async def on_start(self, call_id: str) -> None:
        await self.send(my_tts("Hi, how can I help?"))   # PCM back to the caller
```

`self.send(pcm)` speaks. `self.interrupt()` is barge-in: it drops audio you
already queued so the caller can cut your agent off. Call it the moment you
detect speech, or your agent will talk over people.

### Things that will bite you

- **8kHz mu-law.** The phone network is narrowband. Resample to whatever your
  STT wants, and do not assume 16kHz.
- **`bidirectional="true"`** in the Stream XML. Without it you can listen but
  not speak. It is the single most common mistake.
- **`audioop` is gone in Python 3.13.** If you were going to reach for it to do
  mu-law, use `plivo_audio.py` instead. Ours is pure Python and verified against
  the reference G.711 implementation.
- **Tunnel, not localhost.** Plivo dials in from the internet.

---

## Guardrails

Two services, each covering a different kind of failure.

**Modulate** hears the audio itself: emotion, speaker role, whether the voice is
synthetic, and 164 named behaviors. **Tavily** gives the agent live, cited
knowledge so it does not invent answers.

They map onto tonight's judging criteria directly:

| Judged on | Use | How |
|---|---|---|
| Fraud detection | Modulate | Synthetic voice detection, plus `vishing`, `account_impersonation`, `credential_solicitation` |
| Hallucination prevention | Tavily + Modulate | Ground answers in real sources; `preset:hallucination_policy` catches the agent inventing policy |
| Compliance | Modulate | `recording_consent_omission`, `fdcpa_violation_risk`, `do_not_call_violation_risk` |

See [`src/behaviors.py`](src/behaviors.py) for presets grouped by category, all
confirmed against the live API. `python scripts/list_presets.py` prints all 164
against your own key.

### The part that actually matters

Detecting something is easy. The interesting bit is **feeding it back into the
agent mid-call so its behavior changes**. In [`src/bot.py`](src/bot.py) a
detection becomes a system turn in the same context the agent is reasoning over,
so the agent adapts in its own voice instead of being interrupted by a script.

A dashboard that watches the call is not a guardrail. A loop that changes the
call is.

---

## What we measured, so you do not have to

Run against the live APIs on 5 October 2026. These numbers are the difference
between a demo that works and one that dies on stage.

**Tavily, searching mid-call while the caller waits in silence:**

| `search_depth` | `include_answer` | median | worst |
|---|---|---|---|
| `advanced` | `advanced` | 5.46s | 5.79s |
| `basic` | `basic` | 3.80s | 4.60s |
| **`fast`** | **`advanced`** | **1.27s** | **1.53s** |
| `fast` | `basic` | 0.47s | 1.75s |

Use `fast`. It is a different index, not a truncated `advanced`, and the answer
quality holds up. Anything above about 2 seconds and callers think the line
dropped. If you must do a slow lookup, speak a filler line first.

**Modulate, on a 9.4 second clip:**

| Endpoint | Time |
|---|---|
| `velma-2-stt-batch` | 4.6s |
| `velma-2-synthetic-voice-detection-batch` | 3.4s |

Use the **streaming** endpoints during a live call. The batch ones are for
after.

**Synthetic voice detection works.** We fed it TTS audio and got `synthetic` at
**97%** confidence across every 4-second window, with a fresh verdict about once
a second on the streaming endpoint. A threshold of 0.85 fires reliably without
being twitchy.

### Read this before you plan your build

We timed when each Velma signal actually arrives, on a 9.4s call fed in real
time:

| Signal | Arrives | Usable mid-call? |
|---|---|---|
| Deepfake verdict | ~1s, continuously | **yes** |
| Partial transcript | 3.2s | **yes** |
| Final transcript | 13.3s | yes |
| Topics | 77.4s | no |
| Summary | 78.0s | no |
| Behavior detections | **never, on streaming** | **no** |

**Behaviors are batch-only right now.** We tested twice, with two different
audio samples and two different behavior sets, waiting for the `done` event each
time. The streaming endpoint emitted zero `behavior_detection` events. The batch
endpoint, given identical audio and config, returned:

```
DETECTED Jailbreak Attempt        conf=0.995
DETECTED AI Agent Manipulation    conf=0.990
DETECTED Credential Solicitation  conf=0.995
DETECTED Vishing                  conf=0.980
```

So behaviors work, and work very well. They are just not a live signal today.
Plan accordingly:

- **For a mid-call guardrail**, use the deepfake detector (about 1s) or run your
  own fast check over the live transcript, which you get at ~3s.
- **For compliance and QA**, run Velma batch after the call. 99% confidence with
  written reasoning is a far better audit record than anything you will build in
  80 minutes.

If you find a config flag that turns on streaming behaviors, tell us and we will
update this. Worth asking the Modulate engineer during their session.

---

## Reference agent

[`reference/`](reference/) is a complete agent: Plivo for the call, Modulate for
STT and voice intelligence, Tavily for grounded answers, with guardrails wired
back into the conversation.

There is also a rehearsal mode that replays a scripted call onto the dashboard
with no phone and no API spend, which is handy when the venue wifi gives out:

```bash
python src/server.py
curl -X POST localhost:7860/rehearse
```

---

## Credits and keys

- **Plivo**: you need a number and credits. Ask at the Plivo table.
- **Modulate**: API key and credit code handed out during their 5:55 session.
- **Tavily**: key from [app.tavily.com](https://app.tavily.com).

---

## Licence

MIT. Fork it, ship it, no attribution needed.
