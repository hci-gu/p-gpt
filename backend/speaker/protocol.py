from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ProtocolModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        populate_by_name=True,
    )


class ClientEvent(ProtocolModel):
    event_id: str = Field(min_length=1, max_length=128)


class SpeakerHistoryMessage(ProtocolModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)
    finish_reason: Literal["interrupted"] | None = None


class AudioFormat(ProtocolModel):
    encoding: Literal["pcm_s16le"]
    sample_rate: int
    channels: int
    frame_samples: int | None = None


class SpeakerGenerationSettings(ProtocolModel):
    model: str = Field(min_length=1, max_length=256)
    temperature: float = Field(ge=0, le=2)
    repeat_penalty: Literal[1, 1.1, 1.2]
    seed: int | None = Field(default=None, ge=0, le=9_007_199_254_740_991)
    max_tokens: int = Field(ge=64, le=8192)
    clone_voice: bool = True
    ref_audio: str | None = Field(default=None, max_length=4096)
    num_step: int = Field(ge=22, le=32)
    speed: float = Field(default=0.8, gt=0, le=2)


class SessionConfigureEvent(ClientEvent):
    type: Literal["session.configure"]
    protocol_version: Literal[1]
    persona_id: str = Field(min_length=1, max_length=256)
    persona_name: str = Field(min_length=1, max_length=512)
    instruction_prompt: str = Field(min_length=1, max_length=100_000)
    history: list[SpeakerHistoryMessage] = Field(default_factory=list, max_length=100)
    generation: SpeakerGenerationSettings
    input_audio: AudioFormat
    output_audio: AudioFormat


class TurnEvent(ClientEvent):
    turn_id: str = Field(min_length=1, max_length=128)
    turn_revision: int = Field(ge=0)


class SpeechCandidateEvent(TurnEvent):
    type: Literal["input.speech_candidate"]


class SpeechCandidateCancelledEvent(TurnEvent):
    type: Literal["input.speech_candidate_cancelled"]


class SpeechStartedEvent(TurnEvent):
    type: Literal["input.speech_started"]
    reopened: bool = False


class SpeechSoftEndedEvent(TurnEvent):
    type: Literal["input.speech_soft_ended"]


class InputLimitReachedEvent(TurnEvent):
    type: Literal["input.limit_reached"]


class ResponseEvent(ClientEvent):
    response_generation: int = Field(ge=1)


class ResponseCancelEvent(ClientEvent):
    type: Literal["response.cancel"]
    response_generation: int | None = Field(default=None, ge=1)


class PlaybackSegmentCompletedEvent(ResponseEvent):
    type: Literal["playback.segment_completed"]
    segment_id: str = Field(min_length=1, max_length=128)


class PlaybackResponseCompletedEvent(ResponseEvent):
    type: Literal["playback.response_completed"]


SpeakerClientEvent = Annotated[
    SessionConfigureEvent
    | SpeechCandidateEvent
    | SpeechCandidateCancelledEvent
    | SpeechStartedEvent
    | SpeechSoftEndedEvent
    | InputLimitReachedEvent
    | ResponseCancelEvent
    | PlaybackSegmentCompletedEvent
    | PlaybackResponseCompletedEvent,
    Field(discriminator="type"),
]

_CLIENT_EVENT_ADAPTER = TypeAdapter(SpeakerClientEvent)


def parse_client_event(raw: str) -> SpeakerClientEvent:
    return _CLIENT_EVENT_ADAPTER.validate_python(json.loads(raw))
