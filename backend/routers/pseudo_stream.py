import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from schemas import InitiateRequest
from services.ollama import validate_model
from services.personas import resolve_system_prompt
from services.requests import (
    log_voice_clone_prompt_result,
    run_pseudo_text_pipeline,
    stream_request_from_initiate_request,
)
from services.voice import generate_pseudo_stream_audio, get_or_create_voice_clone_prompt
from state import CANCELLED_REQUEST_DETAIL, PseudoStreamRequestState, pseudo_stream_requests


logger = logging.getLogger("uvicorn.error.p_gpt")
router = APIRouter()


@router.post("/pseudo-stream/initiate-request")
async def initiate_pseudo_stream_request(
    request: InitiateRequest, http_request: Request
) -> dict[str, str]:
    await validate_model(request.model)
    system_prompt = await resolve_system_prompt(
        request.persona_id, request.persona_name, request.instruction_prompt
    )
    request_id = str(uuid4())
    state = PseudoStreamRequestState(stream_request_from_initiate_request(request, system_prompt))
    pseudo_stream_requests[request_id] = state
    if state.request.clone_voice and state.request.ref_audio:
        prompt_task = asyncio.create_task(
            get_or_create_voice_clone_prompt(http_request.app, state.request.ref_audio)
        )
        state.voice_clone_prompt_task = prompt_task
        prompt_task.add_done_callback(lambda task: log_voice_clone_prompt_result(request_id, task))
    state.text_generation_task = asyncio.create_task(run_pseudo_text_pipeline(request_id, state))
    logger.info("Initiated pseudo-stream request: request_id=%s", request_id)
    return {"request_id": request_id}


@router.get("/pseudo-stream/requests/{request_id}/text")
async def get_pseudo_stream_text(request_id: str) -> dict[str, str]:
    state = pseudo_stream_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    await state.text_ready.wait()
    if state.error is not None:
        raise HTTPException(status_code=499 if state.cancelled else 502, detail=state.error)
    if state.generated_text is None:
        raise HTTPException(status_code=502, detail="Text generation failed.")
    return {"request_id": request_id, "generated_text": state.generated_text}


@router.post("/pseudo-stream/requests/{request_id}/interrupt")
async def interrupt_pseudo_stream_request(request_id: str) -> dict[str, str | bool]:
    state = pseudo_stream_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    state.cancelled = True
    state.error = CANCELLED_REQUEST_DETAIL
    state.text_ready.set()
    text_task_cancelled = state.text_generation_task is not None and not state.text_generation_task.done()
    if text_task_cancelled:
        state.text_generation_task.cancel()
    tts_task_cancelled = state.tts_generation_task is not None and not state.tts_generation_task.done()
    if tts_task_cancelled:
        state.tts_generation_task.cancel()
    logger.info("Interrupted pseudo-stream request: request_id=%s text_task_cancelled=%s tts_task_cancelled=%s", request_id, text_task_cancelled, tts_task_cancelled)
    return {
        "interrupted": True, "request_id": request_id,
        "text_generation_task_cancelled": text_task_cancelled,
        "tts_generation_task_cancelled": tts_task_cancelled,
    }


@router.get("/pseudo-stream/requests/{request_id}/audio")
async def stream_pseudo_stream_audio(request_id: str, http_request: Request) -> StreamingResponse:
    state = pseudo_stream_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    if state.audio_started:
        raise HTTPException(status_code=409, detail="Audio streaming has already started for this request.")
    if state.request.response_format != "pcm":
        raise HTTPException(status_code=422, detail="Pseudo-streaming currently requires PCM output.")
    state.audio_started = True
    first_text = await state.sentence_queue.get()
    if first_text is None:
        if state.error is not None:
            raise HTTPException(status_code=499 if state.cancelled else 502, detail=state.error)
        raise HTTPException(status_code=502, detail="Ollama completed without text for speech generation.")
    app = http_request.app
    voice_clone_prompt = None
    if state.request.clone_voice and state.request.ref_audio:
        prompt_task = state.voice_clone_prompt_task
        if prompt_task is None:
            prompt_task = asyncio.create_task(get_or_create_voice_clone_prompt(app, state.request.ref_audio))
            state.voice_clone_prompt_task = prompt_task
        try:
            voice_clone_prompt = await asyncio.shield(prompt_task)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Voice clone preparation failed: {exc}") from exc
    elif state.request.clone_voice:
        voice_clone_prompt = app.state.default_voice_clone_prompt
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)

    async def audio_chunks():
        next_text: str | None = first_text
        chunk_count = 0
        stream_start = perf_counter()
        state.tts_generation_task = asyncio.current_task()
        try:
            while next_text is not None:
                if state.cancelled:
                    break
                audio_bytes = await generate_pseudo_stream_audio(app, next_text, state.request, voice_clone_prompt)
                chunk_count += 1
                logger.info("Yielding pseudo-stream audio chunk: request_id=%s chunk=%s bytes=%s", request_id, chunk_count, len(audio_bytes))
                yield audio_bytes
                next_text = await state.sentence_queue.get()
            if state.error is not None and not state.cancelled:
                raise RuntimeError(state.error)
        except asyncio.CancelledError:
            state.cancelled = True
            state.error = CANCELLED_REQUEST_DETAIL
        except Exception:
            logger.exception("Pseudo-stream audio failed: request_id=%s", request_id)
            raise
        finally:
            state.tts_generation_task = None
            logger.info("Pseudo-stream audio finished: request_id=%s chunks=%s total=%.3fs", request_id, chunk_count, perf_counter() - stream_start)
    return StreamingResponse(audio_chunks(), media_type="audio/pcm")
