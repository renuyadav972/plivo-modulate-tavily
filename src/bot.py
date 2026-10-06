"""The demo agent: Plivo carries the call, Modulate hears it, Tavily grounds it.

Modulate is in the critical path. Velma is the pipeline's speech-to-text
service, so every word the agent reasons over comes from Modulate, and the
emotion, behaviors and topics arrive on that same connection rather than from a
second system bolted alongside.

Its findings are injected back into the same LLM context the agent is already
reasoning over, which is what makes an intervention audible on the call instead
of only visible on the dashboard. See on_guardrail below.

STT_PROVIDER=deepgram falls back to a conventional STT vendor. That exists as
stage insurance, not as the intended configuration.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

import events
import tavily_tools
from modulate import DeepfakeTap, ModulateSTTService, build_guardrail_gate

load_dotenv(override=True)

SYSTEM_PROMPT = (
    "You are the Plivo voice assistant, answering a real phone call. You help "
    "callers understand Plivo, a cloud communications platform offering Voice, "
    "Messaging, Verify, and Voice AI, including SIP trunking called Zentrunk and "
    "audio streaming that connects AI agents to phone calls. "
    "You have two search tools. Use search_the_docs for anything about Plivo "
    "itself, and search_the_web for current news, announcements, or facts you are "
    "unsure of. Search rather than guess. When an answer comes from a search, "
    "say where it came from in a natural spoken way, for example 'according to "
    "the Plivo docs'. "
    "Never invent API fields, pricing, or limits. If a search returns nothing "
    "useful, say you do not know and offer to follow up. "
    "Speak in plain, natural sentences. Keep answers to one to three sentences "
    "and ask one question at a time. Your words become audio, so never use "
    "markdown, lists, or special characters, and say web addresses the way a "
    "person would speak them. "
    "Pronounce the company name Plivo as PLEE-vo."
)


def _create_stt(gate):
    """Modulate Velma is the agent's ears, with a conventional STT as fallback.

    The fallback is reached only by setting STT_PROVIDER=deepgram or by leaving
    MODULATE_API_KEY unset. If it is reached, the Modulate lane on the dashboard
    will stay empty apart from deepfake verdicts, which is the honest signal that
    the demo is running degraded.
    """
    provider = os.getenv("STT_PROVIDER", "modulate").lower()
    modulate_key = os.getenv("MODULATE_API_KEY", "")

    if provider == "modulate" and modulate_key:
        return ModulateSTTService(
            api_key=modulate_key,
            behavior_threshold=float(os.getenv("MODULATE_BEHAVIOR_THRESHOLD", "0.70")),
            gate=gate,
        )

    from pipecat.services.deepgram.stt import DeepgramSTTService

    reason = "STT_PROVIDER is set to deepgram" if provider != "modulate" else "MODULATE_API_KEY is not set"
    logger.warning(f"Falling back to Deepgram for STT because {reason}")
    events.publish("modulate", "error", {"message": f"STT fell back to Deepgram, {reason}"})
    return DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))


def _create_services():
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )
    llm.register_function("search_the_web", tavily_tools.search_the_web)
    llm.register_function("search_the_docs", tavily_tools.search_the_docs)

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv(
            "CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"
        ),
    )

    return llm, tts


async def run_phone_bot(websocket):
    """Run the demo agent for one Plivo call."""
    logger.info("Plivo WebSocket connected, parsing telephony handshake")

    transport_type, call_data = await parse_telephony_websocket(websocket)
    logger.info(f"Provider: {transport_type}, call: {call_data}")

    events.reset()
    events.publish(
        "plivo",
        "stream",
        {
            "provider": transport_type,
            "call_id": call_data.get("call_id"),
            "stream_id": call_data.get("stream_id"),
        },
    )

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    llm, tts = _create_services()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages, tools=tavily_tools.TOOLS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    task: PipelineTask | None = None

    async def on_guardrail(reason: str, guidance: str):
        """Modulate found something, so change what the agent does next.

        The guidance goes in as a system turn rather than a scripted line, which
        keeps the agent's voice consistent while changing its behavior. For a
        confirmed synthetic voice we also interrupt immediately, because waiting
        for the caller to finish defeats the purpose of the check.
        """
        logger.warning(f"Applying guardrail to the live conversation: {reason}")
        messages.append({"role": "system", "content": guidance})

        if task is None:
            return

        if reason == "synthetic_voice":
            await task.queue_frames(
                [
                    TTSSpeakFrame(
                        "Before we go further, I need to pause here and bring in "
                        "a human agent to verify a few details."
                    ),
                    LLMRunFrame(),
                ]
            )
        else:
            await task.queue_frames([LLMRunFrame()])

    # One gate shared by both Modulate connections, so the same alarm cannot be
    # raised twice from two directions.
    gate = build_guardrail_gate(on_guardrail)

    stt = _create_stt(gate)
    deepfake = DeepfakeTap(
        api_key=os.getenv("MODULATE_API_KEY", ""),
        threshold=float(os.getenv("MODULATE_DEEPFAKE_THRESHOLD", "0.85")),
        gate=gate,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            deepfake,  # sees raw caller audio, never blocks the call
            stt,       # Modulate Velma: transcript, emotion, behaviors
            user_aggregator,
            llm,       # reaches for Tavily when it is unsure
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, ws):
        logger.info("Plivo audio stream connected")
        messages.append(
            {
                "role": "system",
                "content": (
                    "Greet the caller in one short sentence, say you are the "
                    "Plivo voice assistant, and ask how you can help."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, ws):
        logger.info("Plivo audio stream disconnected")
        events.publish("call", "end", {"message": "Call ended"})
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
