import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import logging
import os
from time import perf_counter, time
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4
from config import settings

import httpx
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from omnivoice import OmniVoice, VoiceClonePrompt
from pydantic import BaseModel, Field

# FastAPI's development runner configures the Uvicorn logger hierarchy rather
# than the root/module logger. Using a child keeps application INFO messages in
# the same terminal feed as server startup and request logs.
logger = logging.getLogger("uvicorn.error.p_gpt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading OmniVoice on %s", device)

    model = OmniVoice.from_pretrained(
        settings.tts_model,
        device_map=device,
        dtype=dtype,
        # Persona references do not currently include transcripts, so the ASR
        # model is required to construct voice-cloning prompts.
        load_asr=True,
        asr_device=device,
    )
    app.state.tts_model = model
    app.state.tts_lock = asyncio.Lock()
    app.state.voice_clone_prompts = {}
    app.state.voice_clone_prompt_tasks = {}

    logger.info("OmniVoice is online; running warmup inference")
    warmup_start = time()
    await asyncio.to_thread(
        model.generate,
        text="This is a warmup generation. Feel free to discard this output.",
        num_step=26,
        speed=0.8,
    )
    logger.info("OmniVoice warmup took %.2fs", time() - warmup_start)

    try:
        yield
    finally:
        prompt_tasks = list(app.state.voice_clone_prompt_tasks.values())
        if prompt_tasks:
            await asyncio.gather(*prompt_tasks, return_exceptions=True)
        app.state.voice_clone_prompts.clear()
        app.state.voice_clone_prompt_tasks.clear()
        del app.state.tts_model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("OmniVoice shut down and released its model resources")


# Define backend application
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pending_requests: dict[str, "RequestState"] = {}
CANCELLED_REQUEST_DETAIL = "Request interrupted."

OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_TEXT_MODEL = settings.ollama_text_model

OMNIVOICE_TTS_MODEL = settings.tts_model
OMNIVOICE_SAMPLE_RATE = 24_000
POCKETBASE_BASE_URL = os.getenv(
    "POCKETBASE_BASE_URL",
    settings.pocketbase_base_url,
).rstrip("/")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class TextGenerationRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    system_prompt: str = "You are concise assistant. Answer helpfully"
    model: str = OLLAMA_TEXT_MODEL
    temperature: float = 1.0
    top_p: float = 0.95
    repeat_penalty: float = 1.0
    seed: int | None = None
    max_tokens: int = 1024
    think: bool | Literal["low", "medium", "high", "max"] = False
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = None


class StreamTTSRequest(TextGenerationRequest):
    tts_model: str = OMNIVOICE_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "wav"
    voice: str = "casual_male"
    clone_voice: bool = True
    ref_audio: str | None = None
    stream_audio: bool = True
    num_step: int = Field(default=26, gt=0)
    speed: float = Field(default=0.8, gt=0)
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=300, gt=0)
    audio_chunk_size: int = Field(default=8192, gt=0)


class InitiateRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = OLLAMA_TEXT_MODEL
    temperature: float = 1.0
    top_p: float = 0.95
    repeat_penalty: float = 1.0
    seed: int | None = None
    max_tokens: int = 1024
    think: bool | Literal["low", "medium", "high", "max"] = False
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = None
    tts_model: str = OMNIVOICE_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "wav"
    voice: str = "casual_male"
    clone_voice: bool = True
    ref_audio: str | None = None
    stream_audio: bool = True
    num_step: int = Field(default=26, gt=0)
    speed: float = Field(default=0.8, gt=0)
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=300, gt=0)
    audio_chunk_size: int = Field(default=8192, gt=0)


class RequestState:
    def __init__(self, request: StreamTTSRequest) -> None:
        self.request = request
        self.cancelled = False
        self.generated_text: str | None = None
        self.error: str | None = None
        self.text_ready = asyncio.Event()
        self.text_generation_lock = asyncio.Lock()
        self.text_generation_started = False
        self.text_generation_task: asyncio.Task[Any] | None = None
        self.tts_generation_task: asyncio.Task[Any] | None = None
        self.voice_clone_prompt_task: asyncio.Task[VoiceClonePrompt] | None = None


def _content_type_for_audio_format(response_format: str) -> str:
    content_types = {
        "aac": "audio/aac",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "pcm": "audio/pcm",
        "wav": "audio/wav",
    }
    return content_types.get(response_format, "application/octet-stream")


def _build_ollama_chat_payload(request: TextGenerationRequest) -> dict[str, Any]:
    if not request.prompt and not request.messages:
        raise HTTPException(
            status_code=422,
            detail="Provide either prompt or messages.",
        )

    messages = request.messages
    if messages is None:
        messages = [
            ChatMessage(role="system", content=request.system_prompt),
            ChatMessage(role="user", content=request.prompt or ""),
        ]

    payload = {
        "model": request.model,
        "messages": [message.model_dump() for message in messages],
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repeat_penalty": request.repeat_penalty,
            "num_predict": request.max_tokens,
        },
    }
    if request.seed is not None:
        payload["options"]["seed"] = request.seed

    payload["think"] = request.think
    if request.reasoning_effort and request.reasoning_effort != "none":
        payload["think"] = request.reasoning_effort

    return payload


async def _generate_ollama_chat_response(
    request: TextGenerationRequest,
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    payload = _build_ollama_chat_payload(request)
    logger.info("Sending request to Ollama: %s", payload)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return response.json()


def _extract_ollama_response_text(response_data: dict[str, Any]) -> str:
    message = response_data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        content = message["content"].strip()
        if content:
            return content

    if isinstance(response_data.get("response"), str):
        response = response_data["response"].strip()
        if response:
            return response

    done_reason = response_data.get("done_reason")
    detail = "Ollama response did not contain generated text."
    if done_reason == "length":
        detail = "Ollama reached the generation token limit before producing final text."

    raise HTTPException(status_code=502, detail=detail)


async def _prepare_reference_audio(ref_audio: str) -> tuple[torch.Tensor, int]:
    parsed_reference = urlparse(ref_audio)
    parsed_pocketbase = urlparse(POCKETBASE_BASE_URL)
    is_pocketbase_file_path = (
        parsed_reference.scheme in {"http", "https"}
        and bool(parsed_reference.netloc)
        and parsed_reference.path.startswith("/api/files/")
    )
    if not is_pocketbase_file_path:
        raise HTTPException(
            status_code=422,
            detail="Voice reference must be a PocketBase file URL.",
        )

    # The frontend uses PocketBase's externally reachable address, while the
    # backend can fetch the same file over loopback. Rebuild the URL against the
    # configured local origin instead of trusting or requesting the supplied
    # host, which also keeps this endpoint from becoming an SSRF proxy.
    local_reference_url = parsed_pocketbase._replace(
        path=parsed_reference.path,
        params="",
        query=parsed_reference.query,
        fragment="",
    ).geturl()
    logger.info(
        "Loading voice reference from PocketBase: supplied=%s local=%s",
        ref_audio,
        local_reference_url,
    )

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(local_reference_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="PocketBase could not provide the persona audio sample.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not load the persona audio sample from PocketBase.",
        ) from exc

    content_type = response.headers.get("content-type", "audio/wav").split(";", 1)[0]
    if not content_type.startswith("audio/"):
        raise HTTPException(status_code=422, detail="Persona reference is not audio.")

    try:
        audio_array, sample_rate = sf.read(
            BytesIO(response.content),
            dtype="float32",
            always_2d=True,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Persona reference audio could not be decoded.",
        ) from exc

    # OmniVoice expects (channels, samples), while soundfile returns
    # (samples, channels).
    waveform = torch.from_numpy(audio_array.T.copy())
    return waveform, sample_rate


async def _create_voice_clone_prompt(ref_audio: str) -> VoiceClonePrompt:
    reference_audio = await _prepare_reference_audio(ref_audio)
    prompt_start = perf_counter()
    logger.info("Creating VoiceClonePrompt for %s", ref_audio)

    async with app.state.tts_lock:
        prompt = await asyncio.to_thread(
            app.state.tts_model.create_voice_clone_prompt,
            ref_audio=reference_audio,
        )

    logger.info(
        "VoiceClonePrompt created for %s in %.3fs",
        ref_audio,
        perf_counter() - prompt_start,
    )
    return prompt


async def _get_or_create_voice_clone_prompt(ref_audio: str) -> VoiceClonePrompt:
    cached_prompt = app.state.voice_clone_prompts.get(ref_audio)
    if cached_prompt is not None:
        logger.info("Using cached VoiceClonePrompt for %s", ref_audio)
        return cached_prompt

    prompt_task = app.state.voice_clone_prompt_tasks.get(ref_audio)
    if prompt_task is None:
        prompt_task = asyncio.create_task(_create_voice_clone_prompt(ref_audio))
        app.state.voice_clone_prompt_tasks[ref_audio] = prompt_task

    try:
        prompt = await asyncio.shield(prompt_task)
    finally:
        if prompt_task.done():
            app.state.voice_clone_prompt_tasks.pop(ref_audio, None)

    app.state.voice_clone_prompts[ref_audio] = prompt
    return prompt


def _log_voice_clone_prompt_result(
    request_id: str,
    task: asyncio.Task[VoiceClonePrompt],
) -> None:
    if task.cancelled():
        return
    exception = task.exception()
    if exception is not None:
        logger.error(
            "VoiceClonePrompt preparation failed: request_id=%s",
            request_id,
            exc_info=(type(exception), exception, exception.__traceback__),
        )


def _build_tts_payload(
    generated_text: str,
    request: StreamTTSRequest,
    voice_clone_prompt: VoiceClonePrompt | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": generated_text,
        "num_step": request.num_step,
        "speed": request.speed,
    }
    if voice_clone_prompt is not None:
        payload["voice_clone_prompt"] = voice_clone_prompt

    return payload


def _encode_generated_audio(
    audio_array: Any,
    sample_rate: int,
    response_format: str,
) -> bytes:
    output = BytesIO()
    if response_format == "pcm":
        sf.write(
            output,
            audio_array,
            sample_rate,
            format="RAW",
            subtype="PCM_16",
            endian="LITTLE",
        )
    elif response_format == "wav":
        sf.write(
            output,
            audio_array,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
    elif response_format == "mp3":
        sf.write(output, audio_array, sample_rate, format="MP3")
    else:
        raise HTTPException(
            status_code=422,
            detail="OmniVoice output supports pcm, wav, or mp3.",
        )
    return output.getvalue()


def _stream_request_from_initiate_request(request: InitiateRequest) -> StreamTTSRequest:
    return StreamTTSRequest(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        repeat_penalty=request.repeat_penalty,
        seed=request.seed,
        max_tokens=request.max_tokens,
        think=request.think,
        reasoning_effort=request.reasoning_effort,
        tts_model=request.tts_model,
        response_format=request.response_format,
        voice=request.voice,
        clone_voice=request.clone_voice,
        ref_audio=request.ref_audio,
        stream_audio=request.stream_audio,
        num_step=request.num_step,
        speed=request.speed,
        text_generation_timeout_seconds=request.text_generation_timeout_seconds,
        tts_timeout_seconds=request.tts_timeout_seconds,
        audio_chunk_size=request.audio_chunk_size,
    )


async def _get_or_generate_text(
    request_id: str,
    state: RequestState,
    wait_timeout_seconds: float | None = None,
) -> str:
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)

    if state.generated_text is not None:
        return state.generated_text

    if state.error is not None:
        status_code = 499 if state.error == CANCELLED_REQUEST_DETAIL else 502
        raise HTTPException(status_code=status_code, detail=state.error)

    if state.text_generation_lock.locked():
        try:
            await asyncio.wait_for(
                state.text_ready.wait(),
                timeout=wait_timeout_seconds,
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=408,
                detail="Timed out waiting for generated text.",
            ) from exc

        if state.error is not None:
            status_code = 499 if state.error == CANCELLED_REQUEST_DETAIL else 502
            raise HTTPException(status_code=status_code, detail=state.error)
        if state.cancelled:
            raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
        if state.generated_text is None:
            raise HTTPException(status_code=502, detail="Text generation failed.")
        return state.generated_text

    async with state.text_generation_lock:
        if state.cancelled:
            raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)

        if state.generated_text is not None:
            return state.generated_text

        if state.error is not None:
            status_code = 499 if state.error == CANCELLED_REQUEST_DETAIL else 502
            raise HTTPException(status_code=status_code, detail=state.error)

        state.text_generation_started = True
        logger.info(f"Generating text for request_id={request_id}")
        try:
            state.text_generation_task = asyncio.current_task()
            text_response = await _generate_ollama_chat_response(
                state.request,
                timeout_seconds=state.request.text_generation_timeout_seconds,
            )
            if state.cancelled:
                raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
            logger.info(f"Text response for request_id={request_id}: {text_response}")
            state.generated_text = _extract_ollama_response_text(text_response)
        except asyncio.CancelledError as exc:
            state.cancelled = True
            state.error = CANCELLED_REQUEST_DETAIL
            raise HTTPException(
                status_code=499,
                detail=CANCELLED_REQUEST_DETAIL,
            ) from exc
        except HTTPException as exc:
            state.error = str(exc.detail)
            raise
        finally:
            state.text_generation_task = None
            state.text_ready.set()

    if state.generated_text is None:
        raise HTTPException(status_code=502, detail="Text generation failed.")
    logger.info(f"Generated text for request_id={request_id}")
    return state.generated_text


def _interrupt_request_state(request_id: str, state: RequestState) -> dict[str, str | bool]:
    state.cancelled = True
    state.error = CANCELLED_REQUEST_DETAIL
    state.text_ready.set()

    task = state.text_generation_task
    text_task_cancelled = False
    if task is not None and not task.done():
        task.cancel()
        text_task_cancelled = True

    tts_task = state.tts_generation_task
    tts_task_cancelled = False
    if tts_task is not None and not tts_task.done():
        tts_task.cancel()
        tts_task_cancelled = True

    logger.info(
        "interrupted request_id=%s text_generation_started=%s "
        "text_task_cancelled=%s tts_task_cancelled=%s",
        request_id,
        state.text_generation_started,
        text_task_cancelled,
        tts_task_cancelled,
    )

    return {
        "interrupted": True,
        "request_id": request_id,
        "text_generation_task_cancelled": text_task_cancelled,
        "tts_generation_task_cancelled": tts_task_cancelled,
    }



@app.post("/initiate-request")
async def initiate_request(request: InitiateRequest) -> dict[str, str]:
    """Frontend's first point of contact for a message. Stores a conversation request and returns a UUID for the stream endpoint.
    Use the returned `request_id` with
        - `GET /requests/{request_id}/text`
        - `GET /requests/{request_id}/audio`.
    """
    request_id = str(uuid4())
    logger.info(f"Initiating request with ID: {request_id}")
    state = RequestState(_stream_request_from_initiate_request(request))
    pending_requests[request_id] = state

    if state.request.clone_voice and state.request.ref_audio:
        prompt_task = asyncio.create_task(
            _get_or_create_voice_clone_prompt(state.request.ref_audio)
        )
        state.voice_clone_prompt_task = prompt_task
        prompt_task.add_done_callback(
            lambda task: _log_voice_clone_prompt_result(request_id, task)
        )

    return {"request_id": request_id}


@app.get("/requests/{request_id}/text")
async def get_initiated_request_text(request_id: str) -> dict[str, Any]:
    """Generate and return the assistant text for an initiated request.

    The generated text is stored so `GET /requests/{request_id}/audio` can use it.
    If another client already generated the text, this endpoint returns the cached
    result.
    """
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")

    logger.info("Fetching text for request ID: %s", request_id)
    generated_text = await _get_or_generate_text(request_id, state)
    return {"request_id": request_id, "generated_text": generated_text}


@app.post("/requests/{request_id}/interrupt")
async def interrupt_initiated_request(request_id: str) -> dict[str, str | bool]:
    """Interrupt an initiated text/audio request if it is still running."""
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")

    return _interrupt_request_state(request_id, state)


@app.get("/requests/{request_id}/audio")
async def get_initiated_request_audio(request_id: str) -> Response:
    """Generate and return the complete TTS audio for an initiated request.

    If text generation is still running, this endpoint waits up to 30 seconds for
    it to finish. OmniVoice does not expose true real-time audio streaming, so
    the response is sent only after generation completes. PCM responses remain
    compatible with the frontend's 24 kHz PCM player.
    """
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    
    logger.info(f"Generating OmniVoice audio for request ID: {request_id}")

    request = state.request
    generated_text = await _get_or_generate_text(
        request_id,
        state,
        wait_timeout_seconds=30,
    )
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)

    request_start = perf_counter()
    voice_clone_prompt = None
    if request.clone_voice and request.ref_audio:
        prompt_task = state.voice_clone_prompt_task
        if prompt_task is None:
            prompt_task = asyncio.create_task(
                _get_or_create_voice_clone_prompt(request.ref_audio)
            )
            state.voice_clone_prompt_task = prompt_task

        try:
            voice_clone_prompt = await asyncio.shield(prompt_task)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "VoiceClonePrompt preparation failed: request_id=%s",
                request_id,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Voice clone preparation failed: {exc}",
            ) from exc

    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)

    tts_payload = _build_tts_payload(
        generated_text,
        request,
        voice_clone_prompt,
    )

    tts_request_start = perf_counter()
    inference_task: asyncio.Task[list[Any]] | None = None
    try:
        state.tts_generation_task = asyncio.current_task()
        async with app.state.tts_lock:
            inference_task = asyncio.create_task(
                asyncio.to_thread(app.state.tts_model.generate, **tts_payload)
            )
            try:
                generated_audios = await asyncio.wait_for(
                    asyncio.shield(inference_task),
                    timeout=request.tts_timeout_seconds,
                )
            except (asyncio.CancelledError, TimeoutError):
                # PyTorch inference in a worker thread cannot be stopped safely.
                # Keep the model lock until it finishes so another request does
                # not run concurrently against the same model.
                await inference_task
                raise
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="OmniVoice generation timed out.",
        ) from exc
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
        raise HTTPException(
            status_code=502,
            detail=f"OmniVoice returned PCM at an unsupported {sample_rate} Hz.",
        )

    try:
        audio_bytes = _encode_generated_audio(
            generated_audios[0],
            sample_rate,
            request.response_format,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="OmniVoice audio could not be encoded.",
        ) from exc

    total_seconds = perf_counter() - request_start
    logger.info(
        "OmniVoice audio completed: request_id=%s bytes=%s total=%.3fs tts=%.3fs",
        request_id,
        len(audio_bytes),
        total_seconds,
        perf_counter() - tts_request_start,
    )
    return Response(
        content=audio_bytes,
        media_type=_content_type_for_audio_format(request.response_format),
    )
