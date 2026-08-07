import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from config import settings


OMNIVOICE_TTS_MODEL = settings.tts_model
OLLAMA_TEXT_MODEL = settings.ollama_text_model


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
    persona_id: str = Field(min_length=1)
    persona_name: str = Field(min_length=1)
    instruction_prompt: str = Field(min_length=1)
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
    num_step: int = 26
    speed: float = Field(default=0.8, gt=0)
    text_generation_timeout_seconds: float = Field(default=60, gt=0)
    tts_timeout_seconds: float = Field(default=300, gt=0)
    audio_chunk_size: int = Field(default=8192, gt=0)

    @field_validator("temperature")
    @classmethod
    def clamp_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("temperature must be finite")
        return min(2.0, max(0.0, value))

    @field_validator("max_tokens")
    @classmethod
    def clamp_max_tokens(cls, value: int) -> int:
        return min(8192, max(64, value))

    @field_validator("repeat_penalty")
    @classmethod
    def clamp_repeat_penalty(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("repeat_penalty must be finite")
        return min(1.2, max(1.0, value))

    @field_validator("seed")
    @classmethod
    def clamp_seed(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return min(9_007_199_254_740_991, max(0, value))

    @field_validator("num_step")
    @classmethod
    def clamp_num_step(cls, value: int) -> int:
        return min(32, max(22, value))


class EvaluateChatHistoryRequest(BaseModel):
    ollama_model: str = Field(min_length=1, max_length=256)
