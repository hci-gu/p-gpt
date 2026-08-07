import asyncio
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import FastAPI
from omnivoice import VoiceClonePrompt

from config import settings
from schemas import ChatMessage, StreamTTSRequest, TextGenerationRequest
from services.ollama import build_chat_payload, validate_model
from services.personas import resolve_system_prompt
from services.voice import generate_pseudo_stream_audio, get_or_create_voice_clone_prompt


logger = logging.getLogger("uvicorn.error.p_gpt")


@dataclass
class SpeakerApplicationContext:
    generation: Any
    tts_request: StreamTTSRequest
    voice_clone_prompt: VoiceClonePrompt | None


async def configure_session(app: FastAPI, event: Any) -> Any:
    from speaker import SpeakerConfiguredContext

    await validate_model(event.generation.model)
    system_prompt = await resolve_system_prompt(
        event.persona_id, event.persona_name, event.instruction_prompt
    )
    history = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": message.role, "content": message.content}
            for message in event.history
            if message.role != "system"
        ],
    ]
    voice_clone_prompt = None
    if event.generation.clone_voice and event.generation.ref_audio:
        voice_clone_prompt = await get_or_create_voice_clone_prompt(app, event.generation.ref_audio)
    elif event.generation.clone_voice:
        voice_clone_prompt = app.state.default_voice_clone_prompt
    tts_request = StreamTTSRequest(
        messages=[], model=event.generation.model, temperature=event.generation.temperature,
        repeat_penalty=event.generation.repeat_penalty, seed=event.generation.seed,
        max_tokens=event.generation.max_tokens, response_format="pcm",
        clone_voice=event.generation.clone_voice, ref_audio=event.generation.ref_audio,
        num_step=event.generation.num_step, speed=event.generation.speed,
    )
    return SpeakerConfiguredContext(
        application=SpeakerApplicationContext(
            generation=event.generation, tts_request=tts_request, voice_clone_prompt=voice_clone_prompt
        ),
        history=history,
    )


async def transcribe_audio(app: FastAPI, audio: bytes, language: Literal["en", "sv"]) -> str:
    queue_start = perf_counter()
    async with app.state.speaker_asr_lock:
        queue_seconds = perf_counter() - queue_start
        adapter = app.state.speaker_asr_router.adapter_for(language)
        inference_start = perf_counter()
        logger.info(
            "Speaker ASR inference acquired: input_language=%s model=%s audio_seconds=%.3f queue_seconds=%.3f device=%s dtype=%s",
            language, adapter.model_id, len(audio) / (16_000 * 2), queue_seconds,
            adapter.device, getattr(adapter, "dtype", "runtime-managed"),
        )
        inference_task = asyncio.create_task(asyncio.to_thread(adapter.transcribe_pcm16, audio))
        try:
            transcript = await asyncio.shield(inference_task)
            logger.info(
                "Speaker ASR inference finished: input_language=%s model=%s inference_seconds=%.3f chars=%s",
                language, adapter.model_id, perf_counter() - inference_start, len(transcript),
            )
            return transcript
        except asyncio.CancelledError:
            await inference_task
            raise


async def stream_text(context: SpeakerApplicationContext, history: list[dict[str, str]]) -> AsyncIterator[str]:
    generation = context.generation
    request = TextGenerationRequest(
        messages=[ChatMessage(**message) for message in history], model=generation.model,
        temperature=generation.temperature, repeat_penalty=generation.repeat_penalty,
        seed=generation.seed, max_tokens=generation.max_tokens,
    )
    payload = build_chat_payload(request)
    payload["stream"] = True
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    response_part = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Ollama returned an invalid streaming response.") from exc
                message = response_part.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str) and content:
                    yield content


async def synthesize_sentence(
    app: FastAPI, context: SpeakerApplicationContext, sentence: str, metadata: Any
) -> bytes:
    return await generate_pseudo_stream_audio(
        app, sentence, context.tts_request, context.voice_clone_prompt,
        {
            "response_generation": metadata.response_generation,
            "segment_id": metadata.segment_id, "session_id": metadata.session_id,
            "turn_id": metadata.turn_id, "turn_revision": metadata.turn_revision,
        },
    )
