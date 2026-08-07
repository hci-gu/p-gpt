from pydantic import BaseModel


class OllamaModelsResponse(BaseModel):
    models: list[str]
    default_model: str
    used_fallback: bool
