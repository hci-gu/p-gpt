import json
import logging
from typing import Any

import httpx
from fastapi import HTTPException

from config import settings
from schemas import ChatMessage, OllamaModelsResponse, TextGenerationRequest


logger = logging.getLogger("uvicorn.error.p_gpt")


async def get_available_models() -> OllamaModelsResponse:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()

        raw_models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            raise ValueError("Ollama model response did not include a model list.")
        model_names = {
            model_name
            for item in raw_models
            if isinstance(item, dict)
            for model_name in (item.get("model"), item.get("name"))
            if isinstance(model_name, str) and model_name.strip()
        }
        if not model_names:
            raise ValueError("Ollama did not report any installed models.")
        model_names.add(settings.ollama_text_model)
        return OllamaModelsResponse(
            models=sorted(model_names),
            default_model=settings.ollama_text_model,
            used_fallback=False,
        )
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not load installed Ollama models; using configured fallback: %s",
            type(exc).__name__,
        )
        return OllamaModelsResponse(
            models=[settings.ollama_text_model],
            default_model=settings.ollama_text_model,
            used_fallback=True,
        )


async def validate_model(model: str) -> None:
    available = await get_available_models()
    if model not in available.models:
        raise HTTPException(
            status_code=422,
            detail=f"The selected Ollama model is not available: {model}",
        )


def build_chat_payload(request: TextGenerationRequest) -> dict[str, Any]:
    if not request.prompt and not request.messages:
        raise HTTPException(status_code=422, detail="Provide either prompt or messages.")
    messages = request.messages
    if messages is None:
        messages = [
            ChatMessage(role="system", content=request.system_prompt),
            ChatMessage(role="user", content=request.prompt or ""),
        ]
    payload: dict[str, Any] = {
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


async def generate_chat_response(
    request: TextGenerationRequest, timeout_seconds: float = 60
) -> dict[str, Any]:
    payload = build_chat_payload(request)
    logger.info(
        "Sending request to Ollama: model=%s message_count=%s",
        payload["model"],
        len(payload["messages"]),
    )
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat", json=payload
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=exc.response.text
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return response.json()


def extract_response_text(response_data: dict[str, Any]) -> str:
    message = response_data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        content = message["content"].strip()
        if content:
            return content
    if isinstance(response_data.get("response"), str):
        response = response_data["response"].strip()
        if response:
            return response
    detail = "Ollama response did not contain generated text."
    if response_data.get("done_reason") == "length":
        detail = "Ollama reached the generation token limit before producing final text."
    raise HTTPException(status_code=502, detail=detail)
