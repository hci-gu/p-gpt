import asyncio

from fastapi import APIRouter, HTTPException, Request
from uuid import uuid4

from schemas import PersonaPreparationRequest
from services.requests import run_persona_preparation
from state import PersonaPreparationState, persona_preparations


router = APIRouter()


@router.post("/persona-preparations")
async def create_persona_preparation(
    request: PersonaPreparationRequest, http_request: Request
) -> dict[str, str | None]:
    if request.prepare_voice_clone_prompt and not request.audio_sample_url:
        raise HTTPException(status_code=422, detail="A replacement audio sample is required for voice preparation.")
    preparation_id = str(uuid4())
    state = PersonaPreparationState(request)
    persona_preparations[preparation_id] = state
    if request.prepare_system_prompt or request.prepare_voice_clone_prompt:
        state.task = asyncio.create_task(
            run_persona_preparation(http_request.app, preparation_id, state)
        )
    else:
        state.status = "ready"
    return {"id": preparation_id, "status": state.status, "error": state.error}


@router.get("/persona-preparations/{preparation_id}")
async def get_persona_preparation(preparation_id: str) -> dict[str, str | None]:
    state = persona_preparations.get(preparation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown persona preparation.")
    return {"id": preparation_id, "status": state.status, "error": state.error}
