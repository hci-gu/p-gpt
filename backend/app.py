from collections.abc import AsyncIterator
import asyncio
import base64
import logging
import torch
from omnivoice import OmniVoice
import os
from time import time
from time import perf_counter
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4
from contextlib import asynccontextmanager
from config import settings

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading in TTS model OmniVoice using device: {device}")

    model = OmniVoice(
        "k2-fa/OmniVoice",
        device_map=device,
        dtype=torch.bfloat16
    )
    logger.info(f"TTS model is online. Running warmup inference")
    t = time()
    model.generate(
        text="This is a warmup generation feel free to discard this output",
    )
    t_tot = time() - t
    logger.info(f"Warmup infernce took {t_tot:.2f} s")

    yield
    
    # Shutdown
    del model
    logger.info(f"Shutting down application, cleaning up model from GPU")

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

INFERENCE_HOST = settings.inference_host # Spark IP

OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_TEXT_MODEL = settings.ollama_text_model

VLLM_BASE_URL = f"http://{INFERENCE_HOST}:8001/v1"
VLLM_TTS_MODEL = "mistralai/Voxtral-4B-TTS-2603"
POCKETBASE_BASE_URL = os.getenv("POCKETBASE_BASE_URL", "http://127.0.0.1:8090").rstrip("/")


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
    tts_model: str = VLLM_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "wav"
    voice: str = "casual_male"
    clone_voice: bool = True
    ref_audio: str | None = None
    stream_audio: bool = True
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=120, gt=0)
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
    tts_model: str = VLLM_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac", "pcm"] = "wav"
    voice: str = "casual_male"
    clone_voice: bool = True
    ref_audio: str | None = None
    stream_audio: bool = True
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=120, gt=0)
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
    print(f"Sending request to Ollama: {payload}")
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


async def _prepare_reference_audio(ref_audio: str) -> str:
    parsed_reference = urlparse(ref_audio)
    parsed_pocketbase = urlparse(POCKETBASE_BASE_URL)
    is_pocketbase_file = (
        parsed_reference.scheme == parsed_pocketbase.scheme
        and parsed_reference.netloc == parsed_pocketbase.netloc
        and parsed_reference.path.startswith("/api/files/")
    )
    if not is_pocketbase_file:
        raise HTTPException(
            status_code=422,
            detail="Voice reference must be a PocketBase file URL.",
        )

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(ref_audio)
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

    encoded_audio = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{encoded_audio}"


async def _build_tts_payload(
    generated_text: str, request: StreamTTSRequest
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": generated_text,
        "model": request.tts_model,
        "response_format": request.response_format,
    }
    if request.clone_voice and request.ref_audio:
        payload["ref_audio"] = await _prepare_reference_audio(request.ref_audio)
    else:
        payload["voice"] = request.voice

    # vLLM-Omni raw audio streaming is reliable for PCM. Some model/format
    # combinations, including Voxtral WAV in our setup, close chunked responses
    # mid-body when stream_format=audio is requested. Keep WAV browser-playable
    # through the stable non-streaming upstream path until the frontend has a
    # PCM/WebAudio player.
    if request.stream_audio and request.response_format == "pcm":
        payload["stream"] = True
        payload["stream_format"] = "audio"

    return payload


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
    task_cancelled = False
    if task is not None and not task.done():
        task.cancel()
        task_cancelled = True

    logger.info(
        "interrupted request_id=%s text_generation_started=%s task_cancelled=%s",
        request_id,
        state.text_generation_started,
        task_cancelled,
    )

    return {
        "interrupted": True,
        "request_id": request_id,
        "text_generation_task_cancelled": task_cancelled,
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
    pending_requests[request_id] = RequestState(_stream_request_from_initiate_request(request))
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

    print(f"Fetching text for request ID: {request_id}")
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
async def stream_initiated_request_audio(request_id: str) -> StreamingResponse:
    """Stream raw TTS audio bytes for an initiated request.

    If text generation is still running, this endpoint waits up to 30 seconds for
    it to finish. The response body is raw audio bytes, suitable for frontend
    audio playback.
    """
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    
    logger.info(f"Streaming audio for request ID: {request_id}")

    request = state.request
    generated_text = await _get_or_generate_text(
        request_id,
        state,
        wait_timeout_seconds=30,
    )
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)

    request_start = perf_counter()
    tts_payload = await _build_tts_payload(generated_text, request)

    client = httpx.AsyncClient(timeout=request.tts_timeout_seconds)
    tts_request_start = perf_counter()
    stream_context = client.stream(
        "POST",
        f"{VLLM_BASE_URL}/audio/speech",
        json=tts_payload,
    )

    try:
        response = await stream_context.__aenter__()
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        error_body = (await exc.response.aread()).decode(errors="replace")
        await stream_context.__aexit__(type(exc), exc, exc.__traceback__)
        await client.aclose()
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=error_body,
        ) from exc
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def audio_chunks() -> AsyncIterator[bytes]:
        first_chunk_seconds = None
        chunk_count = 0
        total_bytes = 0

        try:
            async for chunk in response.aiter_bytes(chunk_size=request.audio_chunk_size):
                if state.cancelled:
                    logger.info(
                        "initiated audio stream interrupted: request_id=%s",
                        request_id,
                    )
                    break

                if chunk:
                    now = perf_counter()
                    chunk_count += 1
                    total_bytes += len(chunk)

                    if first_chunk_seconds is None:
                        first_chunk_seconds = now - tts_request_start
                        logger.info(
                            "initiated stream first audio chunk: %.3fs after TTS request "
                            "(request_id=%s)",
                            first_chunk_seconds,
                            request_id,
                        )

                    yield chunk
        except httpx.RemoteProtocolError:
            logger.exception(
                "upstream TTS stream ended with an incomplete chunked response: "
                "request_id=%s chunks=%s bytes=%s",
                request_id,
                chunk_count,
                total_bytes,
            )
        finally:
            total_seconds = perf_counter() - request_start
            logger.info(
                "initiated audio stream completed: request_id=%s chunks=%s bytes=%s "
                "total=%.3fs first_tts_chunk=%.3fs",
                request_id,
                chunk_count,
                total_bytes,
                total_seconds,
                first_chunk_seconds or -1,
            )
            await stream_context.__aexit__(None, None, None)
            print(f"Closing HTTP client for request ID: {request_id}")
            await client.aclose()

    return StreamingResponse(
        audio_chunks(),
        media_type=_content_type_for_audio_format(request.response_format),
    )
