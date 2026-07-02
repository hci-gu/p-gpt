from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

app = FastAPI()

REMOTE_HOST = "100.113.76.118"

OLLAMA_BASE_URL = f"http://{REMOTE_HOST}:11434"
OLLAMA_TEXT_MODEL = "google/gemma-4-E4B-it"

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


def _content_type_for_audio_format(response_format: str) -> str:
    content_types = {
        "aac": "audio/aac",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "wav": "audio/wav",
    }
    return content_types.get(response_format, "application/octet-stream")


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

    try:
        async with httpx.AsyncClient(timeout=60) as client:
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
