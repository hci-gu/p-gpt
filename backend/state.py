import asyncio
from typing import Any, Literal

from omnivoice import VoiceClonePrompt

from schemas import PersonaPreparationRequest, StreamTTSRequest


CANCELLED_REQUEST_DETAIL = "Request interrupted."


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
        self.tts_generation_task: asyncio.Task[Any] | None = None
        self.voice_clone_prompt_task: asyncio.Task[VoiceClonePrompt] | None = None


class PseudoStreamRequestState:
    def __init__(self, request: StreamTTSRequest) -> None:
        self.request = request
        self.cancelled = False
        self.error: str | None = None
        self.generated_text: str | None = None
        self.text_ready = asyncio.Event()
        self.sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.text_generation_task: asyncio.Task[Any] | None = None
        self.tts_generation_task: asyncio.Task[Any] | None = None
        self.voice_clone_prompt_task: asyncio.Task[VoiceClonePrompt] | None = None
        self.audio_started = False


class PersonaPreparationState:
    def __init__(self, request: PersonaPreparationRequest) -> None:
        self.request = request
        self.error: str | None = None
        self.status: Literal["pending", "ready", "error"] = "pending"
        self.task: asyncio.Task[None] | None = None


pending_requests: dict[str, RequestState] = {}
pseudo_stream_requests: dict[str, PseudoStreamRequestState] = {}
persona_preparations: dict[str, PersonaPreparationState] = {}
persona_prompt_locks: dict[str, asyncio.Lock] = {}
