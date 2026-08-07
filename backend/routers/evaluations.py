import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import settings
from schemas import EvaluateChatHistoryRequest
from services.evaluations import EVALUATION_STAGES, evaluate_conversation, normalize_conversation


logger = logging.getLogger("uvicorn.error.p_gpt")
router = APIRouter()


def _event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _authorization_header(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="PocketBase authentication is required.")
    return authorization


def _record_url(chat_history_id: str) -> str:
    return f"{settings.pocketbase_base_url.rstrip('/')}/api/collections/chat_history/records/{chat_history_id}"


async def _get_record(chat_history_id: str, authorization: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(_record_url(chat_history_id), headers={"Authorization": authorization})
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if response.status_code in {401, 403}:
        raise HTTPException(status_code=403, detail="You cannot evaluate this conversation.")
    response.raise_for_status()
    record = response.json()
    if not isinstance(record, dict):
        raise HTTPException(status_code=502, detail="PocketBase returned an invalid conversation.")
    return record


async def _update_record(chat_history_id: str, authorization: str, body: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.patch(
            _record_url(chat_history_id), headers={"Authorization": authorization}, json=body
        )
    response.raise_for_status()


@router.get("/evaluation-config")
async def get_evaluation_config() -> dict[str, bool]:
    configured_model = settings.evaluation_model or ""
    return {"cloud_evaluation": bool(configured_model and not configured_model.startswith("ollama:/"))}


@router.post("/chat-history/{chat_history_id}/evaluate")
async def evaluate_chat_history(
    chat_history_id: str, payload: EvaluateChatHistoryRequest, request: Request
) -> StreamingResponse:
    authorization = _authorization_header(request)

    async def stream() -> AsyncIterator[str]:
        evaluation_started = False
        try:
            yield _event("progress", {"stage": "loading", "progress": 2, "message": "Checking conversation"})
            record = await _get_record(chat_history_id, authorization)
            conversation = normalize_conversation(record.get("conversation"))
            if not conversation:
                raise HTTPException(status_code=422, detail="A conversation needs at least one message to be evaluated.")
            if record.get("status", "active") != "active":
                raise HTTPException(status_code=409, detail="This conversation is already being evaluated or is completed.")
            await _update_record(chat_history_id, authorization, {"status": "evaluating"})
            evaluation_started = True
            stage = EVALUATION_STAGES[0]
            yield _event("progress", {"stage": stage[0], "progress": stage[1], "message": stage[2]})

            # The evaluator reports individual judge completions from its worker thread.
            queue: asyncio.Queue[tuple[str, int, str] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def run_evaluation() -> dict[str, Any]:
                try:
                    return evaluate_conversation(
                        conversation,
                        payload.ollama_model,
                        on_stage=lambda stage: loop.call_soon_threadsafe(queue.put_nowait, stage),
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            evaluation_task = asyncio.create_task(asyncio.to_thread(run_evaluation))
            while True:
                stage = await queue.get()
                if stage is None:
                    break
                yield _event("progress", {"stage": stage[0], "progress": stage[1], "message": stage[2]})
            evaluation = await evaluation_task
            saving_stage = EVALUATION_STAGES[-1]
            yield _event("progress", {"stage": saving_stage[0], "progress": saving_stage[1], "message": saving_stage[2]})
            completed_at = datetime.now(UTC).isoformat()
            await _update_record(
                chat_history_id,
                authorization,
                {"status": "completed", "completed_at": completed_at, "evaluation": evaluation},
            )
            yield _event("result", {"progress": 100, "evaluation": evaluation, "completed_at": completed_at})
        except asyncio.CancelledError:
            if evaluation_started:
                await _update_record(chat_history_id, authorization, {"status": "active"})
            raise
        except HTTPException as exc:
            yield _event("error", {"message": str(exc.detail)})
        except Exception:
            logger.exception("Conversation evaluation failed: chat_history_id=%s", chat_history_id)
            if evaluation_started:
                try:
                    await _update_record(chat_history_id, authorization, {"status": "active"})
                except Exception:
                    logger.exception("Failed to restore active status: chat_history_id=%s", chat_history_id)
            yield _event("error", {"message": "Evaluation failed. Please try again."})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
