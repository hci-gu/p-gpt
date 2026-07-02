from collections.abc import AsyncIterator
import asyncio
import logging
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()
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
logger = logging.getLogger(__name__)
pending_requests: dict[str, "RequestState"] = {}

REMOTE_HOST = "100.113.76.118"

OLLAMA_BASE_URL = f"http://{REMOTE_HOST}:11434"
OLLAMA_TEXT_MODEL = "gemma4:e4b"

VLLM_BASE_URL = f"http://{REMOTE_HOST}:8000/v1"
VLLM_TTS_MODEL = "mistralai/Voxtral-4B-TTS-2603"


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
    max_tokens: int = 256
    reasoning_effort: str = "none"


class TTSRequest(BaseModel):
    input: str
    model: str = VLLM_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac"] = "wav"
    voice: str = "casual_male"
    timeout_seconds: float = Field(default=20, gt=0)


class StreamTTSRequest(TextGenerationRequest):
    tts_model: str = VLLM_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac"] = "wav"
    voice: str = "casual_male"
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=120, gt=0)
    audio_chunk_size: int = Field(default=8192, gt=0)


class InitiateRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = OLLAMA_TEXT_MODEL
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    reasoning_effort: str = "none"
    tts_model: str = VLLM_TTS_MODEL
    response_format: Literal["wav", "mp3", "opus", "aac", "flac"] = "wav"
    voice: str = "casual_male"
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=120, gt=0)
    audio_chunk_size: int = Field(default=8192, gt=0)


class RequestState:
    def __init__(self, request: StreamTTSRequest) -> None:
        self.request = request
        self.generated_text: str | None = None
        self.error: str | None = None
        self.text_ready = asyncio.Event()
        self.text_generation_lock = asyncio.Lock()
        self.text_generation_started = False


def _content_type_for_audio_format(response_format: str) -> str:
    content_types = {
        "aac": "audio/aac",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
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
            "num_predict": request.max_tokens,
        },
    }

    if request.reasoning_effort:
        payload["options"]["reasoning_effort"] = request.reasoning_effort

    return payload


async def _generate_ollama_chat_response(
    request: TextGenerationRequest,
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    payload = _build_ollama_chat_payload(request)

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
        return message["content"]

    if isinstance(response_data.get("response"), str):
        return response_data["response"]

    raise HTTPException(
        status_code=502,
        detail="Ollama response did not contain generated text.",
    )


def _build_tts_payload(generated_text: str, request: StreamTTSRequest) -> dict[str, str]:
    return {
        "input": generated_text,
        "model": request.tts_model,
        "response_format": request.response_format,
        "voice": request.voice,
    }


def _stream_request_from_initiate_request(request: InitiateRequest) -> StreamTTSRequest:
    return StreamTTSRequest(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        reasoning_effort=request.reasoning_effort,
        tts_model=request.tts_model,
        response_format=request.response_format,
        voice=request.voice,
        text_generation_timeout_seconds=request.text_generation_timeout_seconds,
        tts_timeout_seconds=request.tts_timeout_seconds,
        audio_chunk_size=request.audio_chunk_size,
    )


async def _get_or_generate_text(
    request_id: str,
    state: RequestState,
    wait_timeout_seconds: float | None = None,
) -> str:
    if state.generated_text is not None:
        return state.generated_text

    if state.error is not None:
        raise HTTPException(status_code=502, detail=state.error)

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
            raise HTTPException(status_code=502, detail=state.error)
        if state.generated_text is None:
            raise HTTPException(status_code=502, detail="Text generation failed.")
        return state.generated_text

    async with state.text_generation_lock:
        if state.generated_text is not None:
            return state.generated_text

        if state.error is not None:
            raise HTTPException(status_code=502, detail=state.error)

        state.text_generation_started = True
        try:
            text_response = await _generate_ollama_chat_response(
                state.request,
                timeout_seconds=state.request.text_generation_timeout_seconds,
            )
            state.generated_text = _extract_ollama_response_text(text_response)
        except HTTPException as exc:
            state.error = str(exc.detail)
            raise
        finally:
            state.text_ready.set()

    if state.generated_text is None:
        raise HTTPException(status_code=502, detail="Text generation failed.")
    logger.info("generated text for request_id=%s", request_id)
    return state.generated_text


@app.get("/")
def read_root():
    return {"status": 200,"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.post("/text-generation")
async def generate_text(request: TextGenerationRequest) -> dict[str, Any]:
    """Generate text through the remote Ollama chat endpoint.

    Typical HTTP request:

    ```http
    POST /text-generation HTTP/1.1
    Host: 127.0.0.1:8000
    Content-Type: application/json

    {
      "prompt": "Give me a one-paragraph vLLM tuning checklist.",
      "temperature": 1.0,
      "top_p": 0.95,
      "max_tokens": 256
    }
    ```

    You can also send an explicit chat history instead of `prompt`:

    ```json
    {
      "messages": [
        {"role": "system", "content": "You are concise assistant. Answer helpfully"},
        {"role": "user", "content": "Give me a one-paragraph vLLM tuning checklist."}
      ]
    }
    ```
    """
    return await _generate_ollama_chat_response(request)


@app.post("/tts")
async def generate_tts(request: TTSRequest) -> Response:
    """Generate speech audio through the remote vLLM TTS endpoint.

    Typical HTTP request:

    ```http
    POST /tts HTTP/1.1
    Host: 127.0.0.1:8000
    Content-Type: application/json
    Accept: audio/wav

    {
      "input": "Hello and welcome. How can I help you today?",
      "voice": "casual_male",
      "response_format": "wav"
    }
    ```

    The response body is the generated audio bytes. Save it directly to a file
    with the same extension as `response_format`, for example `output.wav`.
    """
    payload = request.model_dump(exclude={"timeout_seconds"})

    try:
        async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
            response = await client.post(f"{VLLM_BASE_URL}/audio/speech", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=response.content,
        media_type=_content_type_for_audio_format(request.response_format),
    )


@app.post("/initiate-request")
async def initiate_request(request: InitiateRequest) -> dict[str, str]:
    """Store a conversation request and return a UUID for the stream endpoint.

    Typical HTTP request:

    ```http
    POST /initiate-request HTTP/1.1
    Host: 127.0.0.1:8000
    Content-Type: application/json

    {
      "messages": [
        {"role": "system", "content": "You are concise assistant. Answer helpfully"},
        {"role": "user", "content": "Greet me and ask how you can help."}
      ],
      "voice": "casual_male",
      "response_format": "wav"
    }
    ```

    Use the returned `request_id` with `GET /requests/{request_id}/text` and
    `GET /requests/{request_id}/audio`.
    Requests are held in memory, so they are lost if the backend process restarts.
    """
    request_id = str(uuid4())
    pending_requests[request_id] = RequestState(
        _stream_request_from_initiate_request(request)
    )
    return {"request_id": request_id}


@app.get("/requests/{request_id}/text")
async def get_initiated_request_text(request_id: str) -> dict[str, Any]:
    """Generate and return the assistant text for an initiated request.

    Typical HTTP request:

    ```http
    GET /requests/{request_id}/text HTTP/1.1
    Host: 127.0.0.1:8000
    Accept: application/json
    ```

    The generated text is stored so `GET /requests/{request_id}/audio` can use it.
    If another client already generated the text, this endpoint returns the cached
    result.
    """
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")

    generated_text = await _get_or_generate_text(request_id, state)
    return {"request_id": request_id, "generated_text": generated_text}


@app.get("/requests/{request_id}/audio")
async def stream_initiated_request_audio(request_id: str) -> StreamingResponse:
    """Stream raw TTS audio bytes for an initiated request.

    Typical HTTP request:

    ```http
    GET /requests/{request_id}/audio HTTP/1.1
    Host: 127.0.0.1:8000
    Accept: audio/wav
    ```

    If text generation is still running, this endpoint waits up to 30 seconds for
    it to finish. The response body is raw audio bytes, suitable for frontend
    audio playback.
    """
    state = pending_requests.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")

    request = state.request
    generated_text = await _get_or_generate_text(
        request_id,
        state,
        wait_timeout_seconds=30,
    )
    request_start = perf_counter()
    tts_payload = _build_tts_payload(generated_text, request)

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
            await client.aclose()

    return StreamingResponse(
        audio_chunks(),
        media_type=_content_type_for_audio_format(request.response_format),
    )


@app.post("/stream-tts")
async def stream_tts(request: StreamTTSRequest) -> StreamingResponse:
    """Generate text with Ollama, then stream generated TTS audio from vLLM.

    Typical HTTP request:

    ```http
    POST /stream-tts HTTP/1.1
    Host: 127.0.0.1:8000
    Content-Type: application/json
    Accept: audio/wav

    {
      "prompt": "Greet the user in one friendly sentence.",
      "voice": "casual_male",
      "response_format": "wav"
    }
    ```

    The endpoint waits until Ollama finishes the text response, sends that text
    to `/v1/audio/speech`, then forwards audio bytes as chunks. If the vLLM
    server streams audio incrementally, the client receives those chunks as they
    arrive; otherwise the bytes are forwarded as soon as vLLM returns them.

    To validate streaming behavior without saving audio, call
    `/stream-tts/diagnostics` with the same request body.
    """
    request_start = perf_counter()
    text_response = await _generate_ollama_chat_response(
        request,
        timeout_seconds=request.text_generation_timeout_seconds,
    )
    print(f"Text response: {text_response}")
    text_generation_seconds = perf_counter() - request_start
    generated_text = _extract_ollama_response_text(text_response)
    tts_payload = _build_tts_payload(generated_text, request)

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
                if chunk:
                    now = perf_counter()
                    chunk_count += 1
                    total_bytes += len(chunk)

                    if first_chunk_seconds is None:
                        first_chunk_seconds = now - tts_request_start
                        logger.info(
                            "stream-tts first audio chunk: %.3fs after TTS request "
                            "(%.3fs after original request); text generation took %.3fs",
                            first_chunk_seconds,
                            now - request_start,
                            text_generation_seconds,
                        )

                    yield chunk
        finally:
            total_seconds = perf_counter() - request_start
            logger.info(
                "stream-tts completed: chunks=%s bytes=%s total=%.3fs "
                "first_tts_chunk=%.3fs",
                chunk_count,
                total_bytes,
                total_seconds,
                first_chunk_seconds or -1,
            )
            await stream_context.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(
        audio_chunks(),
        media_type=_content_type_for_audio_format(request.response_format),
    )


@app.post("/stream-tts/diagnostics")
async def stream_tts_diagnostics(request: StreamTTSRequest) -> dict[str, Any]:
    """Measure whether vLLM returns TTS audio incrementally.

    Typical HTTP request:

    ```http
    POST /stream-tts/diagnostics HTTP/1.1
    Host: 127.0.0.1:8000
    Content-Type: application/json

    {
      "prompt": "Write a long spoken answer about healthy sleep habits.",
      "voice": "casual_male",
      "response_format": "wav"
    }
    ```

    The response is JSON with first-chunk timing, total timing, byte counts, and
    per-chunk arrival times. If `first_audio_chunk_seconds_since_tts_start` is
    close to `tts_total_seconds` or `chunk_count` is 1, the upstream TTS server
    likely buffered the audio before responding.
    """
    request_start = perf_counter()
    text_response = await _generate_ollama_chat_response(
        request,
        timeout_seconds=request.text_generation_timeout_seconds,
    )
    text_generation_seconds = perf_counter() - request_start
    generated_text = _extract_ollama_response_text(text_response)
    tts_payload = _build_tts_payload(generated_text, request)

    chunk_events: list[dict[str, float | int]] = []
    total_audio_bytes = 0
    first_audio_chunk_seconds_since_tts_start: float | None = None

    tts_request_start = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=request.tts_timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{VLLM_BASE_URL}/audio/speech",
                json=tts_payload,
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes(
                    chunk_size=request.audio_chunk_size,
                ):
                    if not chunk:
                        continue

                    now = perf_counter()
                    if first_audio_chunk_seconds_since_tts_start is None:
                        first_audio_chunk_seconds_since_tts_start = now - tts_request_start

                    total_audio_bytes += len(chunk)
                    chunk_events.append(
                        {
                            "index": len(chunk_events) + 1,
                            "bytes": len(chunk),
                            "seconds_since_tts_start": round(now - tts_request_start, 3),
                            "seconds_since_request_start": round(now - request_start, 3),
                        }
                    )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    tts_total_seconds = perf_counter() - tts_request_start
    total_seconds = perf_counter() - request_start
    first_chunk_seconds = first_audio_chunk_seconds_since_tts_start
    likely_streaming = (
        first_chunk_seconds is not None
        and len(chunk_events) > 1
        and first_chunk_seconds + 0.25 < tts_total_seconds
    )

    return {
        "likely_streaming": likely_streaming,
        "text_generation_seconds": round(text_generation_seconds, 3),
        "first_audio_chunk_seconds_since_tts_start": (
            round(first_chunk_seconds, 3) if first_chunk_seconds is not None else None
        ),
        "tts_total_seconds": round(tts_total_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "chunk_count": len(chunk_events),
        "total_audio_bytes": total_audio_bytes,
        "audio_chunk_size": request.audio_chunk_size,
        "generated_text": generated_text,
        "chunk_events": chunk_events,
    }
