import asyncio
import logging
from contextlib import asynccontextmanager
from time import perf_counter, time

import soundfile as sf
import torch
from fastapi import FastAPI
from omnivoice import OmniVoice

from config import settings
from services.voice import DEFAULT_VOICE_REFERENCE_PATH, create_default_voice_clone_prompt
from state import persona_preparations


logger = logging.getLogger("uvicorn.error.p_gpt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading OmniVoice on %s", device)
    model = OmniVoice.from_pretrained(
        settings.tts_model, device_map=device, dtype=dtype, load_asr=True, asr_device=device
    )
    app.state.tts_model = model
    app.state.tts_lock = asyncio.Lock()
    app.state.voice_clone_prompts = {}
    app.state.voice_clone_prompt_tasks = {}
    persona_preparations.clear()
    logger.info("OmniVoice is online; running warmup inference")
    warmup_start = time()
    await asyncio.to_thread(
        model.generate, text="This is a warmup generation. Feel free to discard this output.",
        num_step=26, speed=0.8,
    )
    logger.info("OmniVoice warmup took %.2fs", time() - warmup_start)
    logger.info("Computing and caching the default voice clone prompt")
    default_voice_start = perf_counter()
    app.state.default_voice_clone_prompt = await create_default_voice_clone_prompt(app)
    logger.info("Default voice clone prompt computed and cached in %.3fs", perf_counter() - default_voice_start)

    from speaker.asr import KBWhisperASR, ParakeetASR, SpeakerASRRouter

    logger.info("Loading English speaker ASR model %s", settings.speaker_asr_model)
    speaker_asr_start = perf_counter()
    parakeet_asr = await asyncio.to_thread(ParakeetASR.from_pretrained, settings.speaker_asr_model)
    logger.info("Loading Swedish speaker ASR model %s revision=%s", settings.speaker_asr_model_sv, settings.speaker_asr_revision_sv)
    kb_whisper_asr = await asyncio.to_thread(
        KBWhisperASR.from_pretrained, settings.speaker_asr_model_sv, settings.speaker_asr_revision_sv
    )
    speaker_asr_router = SpeakerASRRouter(routes={"en": parakeet_asr, "sv": kb_whisper_asr})
    app.state.speaker_asr_router = speaker_asr_router
    app.state.speaker_asr_lock = asyncio.Lock()
    warmup_audio, warmup_sample_rate = await asyncio.to_thread(
        sf.read, DEFAULT_VOICE_REFERENCE_PATH, dtype="float32", always_2d=True
    )
    for language, adapter in speaker_asr_router.routes.items():
        warmup_start = perf_counter()
        warmup_transcript = await asyncio.to_thread(adapter.transcribe_waveform, warmup_audio, warmup_sample_rate)
        if not warmup_transcript:
            logger.warning("Speaker ASR warmup completed with an empty transcript: language=%s model=%s", language, adapter.model_id)
        logger.info("Speaker ASR route online: language=%s model=%s device=%s dtype=%s warmup_seconds=%.3f", language, adapter.model_id, adapter.device, getattr(adapter, "dtype", "runtime-managed"), perf_counter() - warmup_start)
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        logger.info("Speaker ASR CUDA memory after warmup: allocated_mib=%.1f reserved_mib=%.1f free_mib=%.1f total_mib=%.1f", torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.memory_reserved() / (1024 * 1024), free_bytes / (1024 * 1024), total_bytes / (1024 * 1024))
    logger.info("All speaker ASR routes are online; load and warmup took %.3fs", perf_counter() - speaker_asr_start)
    try:
        yield
    finally:
        preparation_tasks = [state.task for state in persona_preparations.values() if state.task is not None]
        if preparation_tasks:
            await asyncio.gather(*preparation_tasks, return_exceptions=True)
        persona_preparations.clear()
        prompt_tasks = list(app.state.voice_clone_prompt_tasks.values())
        if prompt_tasks:
            await asyncio.gather(*prompt_tasks, return_exceptions=True)
        app.state.voice_clone_prompts.clear()
        app.state.voice_clone_prompt_tasks.clear()
        app.state.speaker_asr_router.close()
        del app.state.speaker_asr_router
        del app.state.speaker_asr_lock
        del app.state.default_voice_clone_prompt
        del app.state.tts_model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("OmniVoice shut down and released its model resources")
