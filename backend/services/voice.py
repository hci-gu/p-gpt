import asyncio
from io import BytesIO
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from omnivoice import VoiceClonePrompt

from config import settings
from schemas import StreamTTSRequest
from speaker.tts_metrics import calculate_speaker_tts_timing


logger = logging.getLogger("uvicorn.error.p_gpt")
OMNIVOICE_SAMPLE_RATE = 24_000
DEFAULT_VOICE_REFERENCE_PATH = Path(__file__).parents[1] / "assets" / "default-voice.mp3"
POCKETBASE_BASE_URL = os.getenv("POCKETBASE_BASE_URL", settings.pocketbase_base_url).rstrip("/")


def content_type_for_audio_format(response_format: str) -> str:
    return {
        "aac": "audio/aac", "flac": "audio/flac", "mp3": "audio/mpeg",
        "opus": "audio/ogg", "pcm": "audio/pcm", "wav": "audio/wav",
    }.get(response_format, "application/octet-stream")


async def prepare_reference_audio(ref_audio: str) -> tuple[torch.Tensor, int]:
    parsed_reference = urlparse(ref_audio)
    parsed_pocketbase = urlparse(POCKETBASE_BASE_URL)
    is_pocketbase_file_path = (
        parsed_reference.scheme in {"http", "https"}
        and bool(parsed_reference.netloc)
        and parsed_reference.path.startswith("/api/files/")
    )
    if not is_pocketbase_file_path:
        raise HTTPException(status_code=422, detail="Voice reference must be a PocketBase file URL.")
    local_reference_url = parsed_pocketbase._replace(
        path=parsed_reference.path, params="", query=parsed_reference.query, fragment=""
    ).geturl()
    logger.info("Loading voice reference from PocketBase: supplied=%s local=%s", ref_audio, local_reference_url)
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(local_reference_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail="PocketBase could not provide the persona audio sample.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not load the persona audio sample from PocketBase.") from exc
    content_type = response.headers.get("content-type", "audio/wav").split(";", 1)[0]
    if not content_type.startswith("audio/"):
        raise HTTPException(status_code=422, detail="Persona reference is not audio.")
    try:
        audio_array, sample_rate = sf.read(BytesIO(response.content), dtype="float32", always_2d=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Persona reference audio could not be decoded.") from exc
    return torch.from_numpy(audio_array.T.copy()), sample_rate


async def prepare_default_voice_reference_audio() -> tuple[torch.Tensor, int]:
    if not DEFAULT_VOICE_REFERENCE_PATH.is_file():
        raise RuntimeError(f"Default voice reference file is missing: {DEFAULT_VOICE_REFERENCE_PATH}")
    try:
        audio_array, sample_rate = await asyncio.to_thread(
            sf.read, DEFAULT_VOICE_REFERENCE_PATH, dtype="float32", always_2d=True
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("Default voice reference audio could not be decoded.") from exc
    return torch.from_numpy(audio_array.T.copy()), sample_rate


async def create_default_voice_clone_prompt(app: FastAPI) -> VoiceClonePrompt:
    reference_audio = await prepare_default_voice_reference_audio()
    async with app.state.tts_lock:
        return await asyncio.to_thread(
            app.state.tts_model.create_voice_clone_prompt, ref_audio=reference_audio
        )


async def create_voice_clone_prompt(app: FastAPI, ref_audio: str) -> VoiceClonePrompt:
    reference_audio = await prepare_reference_audio(ref_audio)
    prompt_start = perf_counter()
    logger.info("Creating VoiceClonePrompt for %s", ref_audio)
    async with app.state.tts_lock:
        prompt = await asyncio.to_thread(
            app.state.tts_model.create_voice_clone_prompt, ref_audio=reference_audio
        )
    logger.info("VoiceClonePrompt created for %s in %.3fs", ref_audio, perf_counter() - prompt_start)
    return prompt


async def get_or_create_voice_clone_prompt(app: FastAPI, ref_audio: str) -> VoiceClonePrompt:
    cached_prompt = app.state.voice_clone_prompts.get(ref_audio)
    if cached_prompt is not None:
        logger.info("Using cached VoiceClonePrompt for %s", ref_audio)
        return cached_prompt
    prompt_task = app.state.voice_clone_prompt_tasks.get(ref_audio)
    if prompt_task is None:
        prompt_task = asyncio.create_task(create_voice_clone_prompt(app, ref_audio))
        app.state.voice_clone_prompt_tasks[ref_audio] = prompt_task
    try:
        prompt = await asyncio.shield(prompt_task)
    finally:
        if prompt_task.done():
            app.state.voice_clone_prompt_tasks.pop(ref_audio, None)
    app.state.voice_clone_prompts[ref_audio] = prompt
    return prompt


def build_tts_payload(
    generated_text: str, request: StreamTTSRequest, voice_clone_prompt: VoiceClonePrompt | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": generated_text, "num_step": request.num_step, "speed": request.speed}
    if voice_clone_prompt is not None:
        payload["voice_clone_prompt"] = voice_clone_prompt
    return payload


async def generate_pseudo_stream_audio(
    app: FastAPI,
    text: str,
    request: StreamTTSRequest,
    voice_clone_prompt: VoiceClonePrompt | None,
    diagnostic_context: dict[str, Any] | None = None,
) -> bytes:
    total_start = perf_counter()
    payload = build_tts_payload(text, request, voice_clone_prompt)
    payload.update({"postprocess_output": False, "pad_duration": 0.0, "fade_duration": 0.0})
    lock_requested_at = perf_counter()
    async with app.state.tts_lock:
        lock_acquired_at = perf_counter()
        inference_task = asyncio.create_task(asyncio.to_thread(app.state.tts_model.generate, **payload))
        try:
            generated_audios = await asyncio.wait_for(
                asyncio.shield(inference_task), timeout=request.tts_timeout_seconds
            )
        except (asyncio.CancelledError, TimeoutError):
            await inference_task
            raise
        inference_finished_at = perf_counter()
    if not generated_audios:
        raise RuntimeError("OmniVoice generated no audio.")
    sample_rate = int(app.state.tts_model.sampling_rate)
    if sample_rate != OMNIVOICE_SAMPLE_RATE:
        raise RuntimeError(f"OmniVoice returned PCM at an unsupported {sample_rate} Hz.")
    encoding_started_at = perf_counter()
    audio_bytes = encode_generated_audio(generated_audios[0], sample_rate, "pcm")
    completed_at = perf_counter()
    timing = calculate_speaker_tts_timing(
        total_started_at=total_start, lock_requested_at=lock_requested_at,
        lock_acquired_at=lock_acquired_at, inference_finished_at=inference_finished_at,
        encoding_started_at=encoding_started_at, completed_at=completed_at,
        output_bytes=len(audio_bytes), sample_rate=sample_rate,
    )
    if diagnostic_context is not None:
        cuda_allocated_mib = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        cuda_reserved_mib = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        timing_arguments = (
            diagnostic_context.get("session_id"), diagnostic_context.get("turn_id"),
            diagnostic_context.get("turn_revision"), diagnostic_context.get("response_generation"),
            diagnostic_context.get("segment_id"), len(text), request.num_step, request.speed,
            timing.lock_wait_seconds, timing.inference_seconds, timing.encoding_seconds,
            timing.total_seconds, timing.audio_seconds, timing.real_time_factor,
            cuda_allocated_mib, cuda_reserved_mib,
        )
        logger.debug("Speaker TTS diagnostics: session=%s turn=%s revision=%s generation=%s segment=%s chars=%s steps=%s speed=%.3f lock_wait=%.3fs inference=%.3fs encoding=%.3fs total=%.3fs audio=%.3fs rtf=%.3f cuda_allocated_mib=%.1f cuda_reserved_mib=%.1f", *timing_arguments)
        if timing.total_seconds > 5 or timing.real_time_factor > 1:
            logger.warning("Speaker TTS slow synthesis: session=%s turn=%s revision=%s generation=%s segment=%s chars=%s steps=%s speed=%.3f lock_wait=%.3fs inference=%.3fs encoding=%.3fs total=%.3fs audio=%.3fs rtf=%.3f cuda_allocated_mib=%.1f cuda_reserved_mib=%.1f", *timing_arguments)
    return audio_bytes


def encode_generated_audio(audio_array: Any, sample_rate: int, response_format: str) -> bytes:
    output = BytesIO()
    if response_format == "pcm":
        sf.write(output, audio_array, sample_rate, format="RAW", subtype="PCM_16", endian="LITTLE")
    elif response_format == "wav":
        sf.write(output, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    elif response_format == "mp3":
        sf.write(output, audio_array, sample_rate, format="MP3")
    else:
        raise HTTPException(status_code=422, detail="OmniVoice output supports pcm, wav, or mp3.")
    return output.getvalue()
