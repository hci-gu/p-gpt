import asyncio
import json
import logging
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from omnivoice import VoiceClonePrompt

from config import settings
from schemas import ChatMessage, InitiateRequest, StreamTTSRequest
from services.ollama import build_chat_payload, extract_response_text, generate_chat_response
from services.personas import resolve_system_prompt
from services.voice import get_or_create_voice_clone_prompt
from state import (
    CANCELLED_REQUEST_DETAIL,
    PersonaPreparationState,
    PseudoStreamRequestState,
    RequestState,
)


logger = logging.getLogger("uvicorn.error.p_gpt")
_SENTENCE_BOUNDARY = re.compile(r'[.!?]+(?:["\')\]]+)?(?=\s)')
_NON_TERMINAL_ABBREVIATIONS = {"dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs"}


def stream_request_from_initiate_request(request: InitiateRequest, system_prompt: str) -> StreamTTSRequest:
    conversation_messages = [
        ChatMessage(role="system", content=system_prompt),
        *[message for message in request.messages if message.role != "system"],
    ]
    return StreamTTSRequest(
        messages=conversation_messages, model=request.model, temperature=request.temperature,
        top_p=request.top_p, repeat_penalty=request.repeat_penalty, seed=request.seed,
        max_tokens=request.max_tokens, think=request.think, reasoning_effort=request.reasoning_effort,
        tts_model=request.tts_model, response_format=request.response_format, voice=request.voice,
        clone_voice=request.clone_voice, ref_audio=request.ref_audio, stream_audio=request.stream_audio,
        num_step=request.num_step, speed=request.speed,
        text_generation_timeout_seconds=request.text_generation_timeout_seconds,
        tts_timeout_seconds=request.tts_timeout_seconds, audio_chunk_size=request.audio_chunk_size,
    )


async def get_or_generate_text(request_id: str, state: RequestState, wait_timeout_seconds: float | None = None) -> str:
    if state.cancelled:
        raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
    if state.generated_text is not None:
        return state.generated_text
    if state.error is not None:
        raise HTTPException(status_code=499 if state.error == CANCELLED_REQUEST_DETAIL else 502, detail=state.error)
    if state.text_generation_lock.locked():
        try:
            await asyncio.wait_for(state.text_ready.wait(), timeout=wait_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(status_code=408, detail="Timed out waiting for generated text.") from exc
        if state.error is not None:
            raise HTTPException(status_code=499 if state.error == CANCELLED_REQUEST_DETAIL else 502, detail=state.error)
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
            raise HTTPException(status_code=499 if state.error == CANCELLED_REQUEST_DETAIL else 502, detail=state.error)
        state.text_generation_started = True
        logger.info("Generating text for request_id=%s", request_id)
        try:
            state.text_generation_task = asyncio.current_task()
            text_response = await generate_chat_response(
                state.request, timeout_seconds=state.request.text_generation_timeout_seconds
            )
            if state.cancelled:
                raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL)
            logger.info("Text response for request_id=%s: %s", request_id, text_response)
            state.generated_text = extract_response_text(text_response)
        except asyncio.CancelledError as exc:
            state.cancelled = True
            state.error = CANCELLED_REQUEST_DETAIL
            raise HTTPException(status_code=499, detail=CANCELLED_REQUEST_DETAIL) from exc
        except HTTPException as exc:
            state.error = str(exc.detail)
            raise
        finally:
            state.text_generation_task = None
            state.text_ready.set()
    if state.generated_text is None:
        raise HTTPException(status_code=502, detail="Text generation failed.")
    logger.info("Generated text for request_id=%s", request_id)
    return state.generated_text


def interrupt_request_state(request_id: str, state: RequestState) -> dict[str, str | bool]:
    state.cancelled = True
    state.error = CANCELLED_REQUEST_DETAIL
    state.text_ready.set()
    task = state.text_generation_task
    text_task_cancelled = task is not None and not task.done()
    if text_task_cancelled:
        task.cancel()
    tts_task = state.tts_generation_task
    tts_task_cancelled = tts_task is not None and not tts_task.done()
    if tts_task_cancelled:
        tts_task.cancel()
    logger.info("interrupted request_id=%s text_generation_started=%s text_task_cancelled=%s tts_task_cancelled=%s", request_id, state.text_generation_started, text_task_cancelled, tts_task_cancelled)
    return {
        "interrupted": True, "request_id": request_id,
        "text_generation_task_cancelled": text_task_cancelled,
        "tts_generation_task_cancelled": tts_task_cancelled,
    }


def is_non_terminal_period(text: str, boundary_start: int) -> bool:
    if text[boundary_start] != ".":
        return False
    word_match = re.search(r"([A-Za-z.]+)\.$", text[: boundary_start + 1])
    if word_match is None:
        return False
    word = word_match.group(1)
    return word.lower() in _NON_TERMINAL_ABBREVIATIONS or (len(word) == 1 and word.isupper())


def extract_complete_sentences(text: str) -> tuple[list[str], str]:
    sentences: list[str] = []
    sentence_start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        if is_non_terminal_period(text, boundary.start()):
            continue
        sentence = text[sentence_start:boundary.end()].strip()
        if sentence:
            sentences.append(sentence)
        sentence_start = boundary.end()
    return sentences, text[sentence_start:].lstrip()


async def run_pseudo_text_pipeline(request_id: str, state: PseudoStreamRequestState) -> None:
    payload = build_chat_payload(state.request)
    payload["stream"] = True
    text_parts: list[str] = []
    sentence_buffer = ""
    pending_sentences: list[str] = []
    logger.info("Starting streamed Ollama response: request_id=%s", request_id)
    try:
        async with httpx.AsyncClient(timeout=state.request.text_generation_timeout_seconds) as client:
            async with client.stream(
                "POST", f"{settings.ollama_base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if state.cancelled:
                        raise asyncio.CancelledError
                    if not line:
                        continue
                    try:
                        response_part = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("Ollama returned an invalid streaming response.") from exc
                    message = response_part.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    if not isinstance(content, str) or not content:
                        continue
                    text_parts.append(content)
                    sentence_buffer += content
                    complete_sentences, sentence_buffer = extract_complete_sentences(sentence_buffer)
                    pending_sentences.extend(complete_sentences)
                    while len(pending_sentences) >= 2:
                        tts_text = " ".join(pending_sentences[:2]).strip()
                        del pending_sentences[:2]
                        if tts_text:
                            await state.sentence_queue.put(tts_text)
                            logger.info("Queued two sentences for pseudo-stream TTS: request_id=%s characters=%s", request_id, len(tts_text))
        final_text = "".join(text_parts).strip()
        if not final_text:
            raise RuntimeError("Ollama response did not contain generated text.")
        remaining_text = " ".join([*pending_sentences, sentence_buffer.strip()]).strip()
        if remaining_text:
            await state.sentence_queue.put(remaining_text)
        state.generated_text = final_text
        logger.info("Streamed Ollama response completed: request_id=%s", request_id)
    except asyncio.CancelledError:
        state.cancelled = True
        state.error = CANCELLED_REQUEST_DETAIL
    except httpx.HTTPStatusError as exc:
        state.error = str(exc)
        logger.exception("Streamed Ollama request failed: request_id=%s", request_id)
    except Exception as exc:
        state.error = str(exc)
        logger.exception("Streamed Ollama request failed: request_id=%s", request_id)
    finally:
        state.text_generation_task = None
        state.text_ready.set()
        await state.sentence_queue.put(None)


async def run_persona_preparation(app: FastAPI, preparation_id: str, state: PersonaPreparationState) -> None:
    request = state.request
    try:
        tasks: list[Any] = []
        if request.prepare_system_prompt:
            tasks.append(resolve_system_prompt(request.persona_id, request.persona_name, request.instruction_prompt))
        if request.prepare_voice_clone_prompt:
            if not request.audio_sample_url:
                raise ValueError("A replacement audio sample is required.")
            previous_audio_url = request.previous_audio_sample_url
            if previous_audio_url and previous_audio_url != request.audio_sample_url:
                app.state.voice_clone_prompts.pop(previous_audio_url, None)
            tasks.append(get_or_create_voice_clone_prompt(app, request.audio_sample_url))
        if tasks:
            await asyncio.gather(*tasks)
        state.status = "ready"
        logger.info("Persona preparation completed: preparation_id=%s", preparation_id)
    except Exception as exc:
        state.error = str(exc)
        state.status = "error"
        logger.exception("Persona preparation failed: preparation_id=%s", preparation_id)
    finally:
        state.task = None


def log_voice_clone_prompt_result(request_id: str, task: asyncio.Task[VoiceClonePrompt]) -> None:
    if task.cancelled():
        return
    exception = task.exception()
    if exception is not None:
        logger.error("VoiceClonePrompt preparation failed: request_id=%s", request_id, exc_info=(type(exception), exception, exception.__traceback__))
