import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from schemas import InitiateRequest
from services.ollama import validate_model
from services.personas import resolve_system_prompt
from services.requests import (
    get_or_generate_text,
    interrupt_request_state,
    log_voice_clone_prompt_result,
    stream_request_from_initiate_request,
)
from services.voice import (
    OMNIVOICE_SAMPLE_RATE,
    build_tts_payload,
    content_type_for_audio_format,
    encode_generated_audio,
    get_or_create_voice_clone_prompt,
)
from state import CANCELLED_REQUEST_DETAIL, RequestState, pending_requests


logger = logging.getLogger("uvicorn.error.p_gpt")
router = APIRouter()


@router.post("/initiate-request")
async def initiate_request(request: InitiateRequest, http_request: Request) -> dict[str, str]:
    """Store a conversation request and return the ID used by text and audio endpoints."""
    await validate_model(request.model)
    system_prompt = await resolve_system_prompt(
        request.persona_id, request.persona_name, request.instruction_prompt
    )
    request_id = str(uuid4())
    logger.info("Initiating request with ID: %s", request_id)
    state = RequestState(stream_request_from_initiate_request(request, system_prompt))
    pending_requests[request_id] = state
    if state.request.clone_voice and state.request.ref_audio:
        prompt_task = asyncio.create_task(
            get_or_create_voice_clone_prompt(http_request.app, state.request.ref_audio)
        )
        state.voice_clone_prompt_task = prompt_task
        prompt_task.add_done_callback(lambda task: log_voice_clone_prompt_result(request_id, task))
    return {"request_id": request_id}


@router.get("/requests/{request_id}/text")
async def get_initiated_request_text(request_id: str) -> dict[str, object]:
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    logger.info("Fetching text for request ID: %s", request_id)
    generated_text = await get_or_generate_text(request_id, state)
    return {"request_id": request_id, "generated_text": generated_text}


@router.post("/requests/{request_id}/interrupt")
async def interrupt_initiated_request(request_id: str) -> dict[str, str | bool]:
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    return interrupt_request_state(request_id, state)


@router.get("/requests/{request_id}/audio")
async def get_initiated_request_audio(request_id: str, http_request: Request) -> Response:
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    app = http_request.app
    logger.info("Generating OmniVoice audio for request ID: %s", request_id)
    request = state.request
    generated_text = await get_or_generate_text(request_id, state, wait_timeout_seconds=30)
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
    request_start = perf_counter()
    voice_clone_prompt = None
    if request.clone_voice and request.ref_audio:
        prompt_task = state.voice_clone_prompt_task
        if prompt_task is None:
            prompt_task = asyncio.create_task(get_or_create_voice_clone_prompt(app, request.ref_audio))
            state.voice_clone_prompt_task = prompt_task
        try:
            voice_clone_prompt = await asyncio.shield(prompt_task)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("VoiceClonePrompt preparation failed: request_id=%s", request_id)
            raise HTTPException(status_code=502, detail=f"Voice clone preparation failed: {exc}") from exc
    elif request.clone_voice:
        voice_clone_prompt = app.state.default_voice_clone_prompt
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
    tts_payload = build_tts_payload(generated_text, request, voice_clone_prompt)
    tts_request_start = perf_counter()
    inference_task = None
    try:
        state.tts_generation_task = asyncio.current_task()
        async with app.state.tts_lock:
            inference_task = asyncio.create_task(asyncio.to_thread(app.state.tts_model.generate, **tts_payload))
            try:
                generated_audios = await asyncio.wait_for(
                    asyncio.shield(inference_task), timeout=request.tts_timeout_seconds
                )
            except (asyncio.CancelledError, TimeoutError):
                await inference_task
                raise
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="OmniVoice generation timed out.") from exc
    except asyncio.CancelledError as exc:
        state.cancelled = True
        state.error = CANCELLED_REQUEST_DETAIL
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL) from exc
    except Exception as exc:
        logger.exception("OmniVoice generation failed: request_id=%s", request_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        state.tts_generation_task = None
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
    if not generated_audios:
        raise HTTPException(status_code=502, detail="OmniVoice generated no audio.")
    sample_rate = int(app.state.tts_model.sampling_rate)
    if request.response_format == "pcm" and sample_rate != OMNIVOICE_SAMPLE_RATE:
        raise HTTPException(status_code=502, detail=f"OmniVoice returned PCM at an unsupported {sample_rate} Hz.")
    try:
        audio_bytes = encode_generated_audio(generated_audios[0], sample_rate, request.response_format)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="OmniVoice audio could not be encoded.") from exc
    logger.info(
        "OmniVoice audio completed: request_id=%s bytes=%s total=%.3fs tts=%.3fs",
        request_id, len(audio_bytes), perf_counter() - request_start, perf_counter() - tts_request_start,
    )
    return Response(content=audio_bytes, media_type=content_type_for_audio_format(request.response_format))
