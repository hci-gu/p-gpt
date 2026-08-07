from fastapi import APIRouter

from schemas import OllamaModelsResponse
from services.ollama import get_available_models


router = APIRouter()


@router.get("/ollama/models")
async def get_ollama_models() -> OllamaModelsResponse:
    """Return conversation models available through the configured Ollama server."""
    return await get_available_models()
