from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import json
import logging
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import ValidationError

from .protocol import (
    InputLimitReachedEvent,
    PlaybackResponseCompletedEvent,
    PlaybackSegmentCompletedEvent,
    ResponseCancelEvent,
    SessionConfigureEvent,
    SessionUpdateEvent,
    SpeakerClientEvent,
    SpeakerInputLanguage,
    SpeechCandidateCancelledEvent,
    SpeechCandidateEvent,
    SpeechSoftEndedEvent,
    SpeechStartedEvent,
    VadDiagnosticEvent,
    parse_client_event,
)

INPUT_FRAME_BYTES = 1_024
MAX_UTTERANCE_BYTES = 60 * 16_000 * 2
OUTPUT_CHUNK_BYTES = 24_000 * 2 // 10
REOPEN_GRACE_SECONDS = 2.0


@dataclass
class SpeakerConfiguredContext:
    application: Any
    history: list[dict[str, str]]


@dataclass(frozen=True)
class SpeakerSynthesisMetadata:
    session_id: str
    turn_id: str
    turn_revision: int
    response_generation: int
    segment_id: str


@dataclass
class SpeakerServices:
    configure: Callable[[SessionConfigureEvent], Awaitable[SpeakerConfiguredContext]]
    transcribe: Callable[[bytes, SpeakerInputLanguage], Awaitable[str]]
    stream_text: Callable[
        [Any, list[dict[str, str]]],
        AsyncIterator[str],
    ]
    synthesize: Callable[[Any, str, SpeakerSynthesisMetadata], Awaitable[bytes]]


class SentenceSegmenter:
    def __init__(self, maximum_characters: int = 180) -> None:
        self.buffer = ""
        self.maximum_characters = maximum_characters

    @staticmethod
    def _is_abbreviation(text: str) -> bool:
        token = text.rstrip().rsplit(" ", 1)[-1].rstrip(".").lower()
        return token in {
            "dr",
            "e.g",
            "etc",
            "i.e",
            "jr",
            "mr",
            "mrs",
            "ms",
            "prof",
            "sr",
            "st",
            "vs",
        }

    def feed(self, value: str) -> list[str]:
        self.buffer += value
        output: list[str] = []
        start = 0
        index = 0
        while index < len(self.buffer):
            if self.buffer[index] not in ".!?…":
                index += 1
                continue
            end = index + 1
            while end < len(self.buffer) and self.buffer[end] in '.!?…"\')]':
                end += 1
            # LLM token boundaries can split an ellipsis. Keep trailing
            # punctuation buffered until the next token proves the boundary.
            if end == len(self.buffer):
                break
            if end < len(self.buffer) and not self.buffer[end].isspace():
                index = end
                continue
            candidate = self.buffer[start:end].strip()
            if self.buffer[index] == "." and self._is_abbreviation(candidate):
                index = end
                continue
            # Never enqueue punctuation by itself. Retaining it in the buffer
            # attaches a standalone ellipsis to the following spoken phrase.
            if candidate and not any(character.isalnum() for character in candidate):
                index = end
                continue
            if candidate:
                output.append(candidate)
            start = end
            index = end

        self.buffer = self.buffer[start:].lstrip()
        while len(self.buffer) >= self.maximum_characters:
            boundary = self.buffer.rfind(" ", 0, self.maximum_characters + 1)
            if boundary <= 0:
                boundary = self.maximum_characters
            output.append(self.buffer[:boundary].strip())
            self.buffer = self.buffer[boundary:].lstrip()
        return [part for part in output if part]

    def finish(self) -> list[str]:
        remaining = self.buffer.strip()
        self.buffer = ""
        return [remaining] if remaining and any(character.isalnum() for character in remaining) else []


class SpeakerSession:
    def __init__(
        self,
        websocket: Any,
        services: SpeakerServices,
        logger: logging.Logger,
        reopen_grace_seconds: float = REOPEN_GRACE_SECONDS,
    ) -> None:
        self.websocket = websocket
        self.services = services
        self.logger = logger
        self.reopen_grace_seconds = reopen_grace_seconds
        self.session_id = str(uuid4())
        self.state: Literal[
            "awaiting_config", "idle", "capturing", "grace", "responding", "closed"
        ] = "awaiting_config"
        self.context: SpeakerConfiguredContext | None = None
        self.input_language: SpeakerInputLanguage = "en"
        self.turn_language: SpeakerInputLanguage | None = None
        self.turn_id: str | None = None
        self.turn_revision = 0
        self.audio = bytearray()
        self.response_generation = 0
        self.current_generation: int | None = None
        self.pipeline_task: asyncio.Task[None] | None = None
        self.grace_deadline = 0.0
        self.grace_held = False
        self.grace_changed = asyncio.Event()
        self.playback_completed = asyncio.Event()
        self.audio_done = False
        self.segment_order: list[str] = []
        self.segment_text: dict[str, str] = {}
        self.played_segments: set[str] = set()
        self.send_lock = asyncio.Lock()
        self.first_audio_sent = False
        self.received_audio_frames = 0

    async def run(self) -> None:
        self.logger.debug("Speaker socket opened: session=%s", self.session_id)
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_bytes = message.get("bytes")
                raw_text = message.get("text")
                if raw_bytes is not None:
                    await self._handle_audio(raw_bytes)
                elif raw_text is not None:
                    await self._handle_text(raw_text)
        finally:
            self.logger.debug(
                "Speaker socket closed: session=%s state=%s turn=%s revision=%s generation=%s",
                self.session_id,
                self.state,
                self.turn_id,
                self.turn_revision,
                self.current_generation,
            )
            self.state = "closed"
            self._invalidate_pipeline()

    async def _handle_text(self, raw: str) -> None:
        try:
            event = parse_client_event(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            await self._send_error("invalid_event", str(exc), fatal=True)
            await self.websocket.close(code=1008)
            self.state = "closed"
            return

        self.logger.debug(
            "Speaker control received: session=%s state=%s type=%s turn=%s revision=%s generation=%s",
            self.session_id,
            self.state,
            event.type,
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
            getattr(event, "response_generation", None),
        )

        if isinstance(event, SessionConfigureEvent):
            await self._configure(event)
        elif self.state == "awaiting_config":
            await self._send_error("session_not_configured", "Configure the session first.")
        elif isinstance(event, SessionUpdateEvent):
            await self._update_session(event)
        elif isinstance(event, SpeechCandidateEvent):
            await self._speech_candidate(event)
        elif isinstance(event, SpeechCandidateCancelledEvent):
            await self._speech_candidate_cancelled(event)
        elif isinstance(event, SpeechStartedEvent):
            await self._speech_started(event)
        elif isinstance(event, (SpeechSoftEndedEvent, InputLimitReachedEvent)):
            await self._speech_soft_ended(event)
        elif isinstance(event, PlaybackSegmentCompletedEvent):
            await self._segment_completed(event)
        elif isinstance(event, PlaybackResponseCompletedEvent):
            await self._response_playback_completed(event)
        elif isinstance(event, ResponseCancelEvent):
            await self._cancel_response("client_cancelled")
        elif isinstance(event, VadDiagnosticEvent):
            self.logger.debug(
                "Speaker client VAD: session=%s state=%s client_phase=%s activity=%s detail=%s epoch=%s profile=%s recovery=%s pending=%s samples=%s probability_min=%s probability_average=%s probability_max=%s processing_average_ms=%s processing_maximum_ms=%s queue_average_ms=%s queue_maximum_ms=%s",
                self.session_id,
                self.state,
                event.phase,
                event.activity,
                event.detail,
                event.capture_epoch,
                event.detection_profile,
                event.recovery_count,
                event.pending_frame_count,
                event.sample_count,
                event.probability_min,
                event.probability_average,
                event.probability_max,
                event.processing_average_milliseconds,
                event.processing_maximum_milliseconds,
                event.queue_delay_average_milliseconds,
                event.queue_delay_maximum_milliseconds,
            )

    async def _configure(self, event: SessionConfigureEvent) -> None:
        if self.state != "awaiting_config":
            await self._send_error("already_configured", "Session is already configured.")
            return
        if (
            event.input_audio.encoding != "pcm_s16le"
            or event.input_audio.sample_rate != 16_000
            or event.input_audio.channels != 1
            or event.input_audio.frame_samples != 512
            or event.output_audio.encoding != "pcm_s16le"
            or event.output_audio.sample_rate != 24_000
            or event.output_audio.channels != 1
        ):
            await self._send_error("unsupported_audio", "Speaker audio format is fixed for protocol v1.", fatal=True)
            await self.websocket.close(code=1008)
            self.state = "closed"
            return
        try:
            self.context = await self.services.configure(event)
        except Exception as exc:
            self.logger.exception("Speaker session configuration failed: session=%s", self.session_id)
            await self._send_error("configuration_failed", str(exc), fatal=True)
            await self.websocket.close(code=1011)
            self.state = "closed"
            return
        self.state = "idle"
        self.input_language = event.input_language
        self.logger.info(
            "Speaker session ready: session=%s history_messages=%s grace_seconds=%.3f input_language=%s",
            self.session_id,
            len(self.context.history),
            self.reopen_grace_seconds,
            self.input_language,
        )
        await self._send_json(
            "session.ready",
            inputLanguage=self.input_language,
            inputAudio={"encoding": "pcm_s16le", "sampleRate": 16_000, "channels": 1, "frameSamples": 512},
            outputAudio={"encoding": "pcm_s16le", "sampleRate": 24_000, "channels": 1},
        )

    async def _update_session(self, event: SessionUpdateEvent) -> None:
        if self.state != "idle":
            await self._send_error(
                "session_busy",
                "The input language can only be changed between turns.",
            )
            return
        previous_language = self.input_language
        self.input_language = event.input_language
        self.logger.info(
            "Speaker input language updated: session=%s previous_language=%s input_language=%s",
            self.session_id,
            previous_language,
            self.input_language,
        )
        await self._send_json(
            "session.updated",
            inputLanguage=self.input_language,
        )

    async def _handle_audio(self, audio: bytes) -> None:
        if self.state != "capturing":
            await self._send_error("audio_outside_capture", "Binary audio is only accepted during capture.")
            return
        if len(audio) != INPUT_FRAME_BYTES:
            await self._send_error("invalid_audio_frame", "Each input frame must contain exactly 1,024 bytes.")
            return
        if len(self.audio) + len(audio) > MAX_UTTERANCE_BYTES:
            await self._send_error("utterance_too_long", "The 60-second utterance limit was reached.")
            await self._start_generation()
            return
        self.audio.extend(audio)
        self.received_audio_frames += 1
        if self.received_audio_frames % 32 == 0:
            self.logger.debug(
                "Speaker audio capture progress: session=%s turn=%s revision=%s frames=%s audio_seconds=%.3f",
                self.session_id,
                self.turn_id,
                self.turn_revision,
                self.received_audio_frames,
                len(self.audio) / (16_000 * 2),
            )

    async def _speech_candidate(self, event: SpeechCandidateEvent) -> None:
        if self.state != "grace" or not self._matches_turn(event):
            self.logger.debug(
                "Speaker candidate ignored: session=%s state=%s turn=%s revision=%s",
                self.session_id,
                self.state,
                event.turn_id,
                event.turn_revision,
            )
            return
        self.grace_held = True
        self.grace_changed.set()
        self.logger.debug(
            "Speaker grace held by speech candidate: session=%s turn=%s revision=%s remaining=%.3fs",
            self.session_id,
            self.turn_id,
            self.turn_revision,
            max(0, self.grace_deadline - asyncio.get_running_loop().time()),
        )

    async def _speech_candidate_cancelled(
        self, event: SpeechCandidateCancelledEvent
    ) -> None:
        if self.state != "grace" or not self._matches_turn(event):
            self.logger.debug(
                "Speaker candidate cancellation ignored: session=%s state=%s turn=%s revision=%s",
                self.session_id,
                self.state,
                event.turn_id,
                event.turn_revision,
            )
            return
        self.grace_held = False
        self.grace_changed.set()
        self.logger.debug(
            "Speaker grace candidate released: session=%s turn=%s revision=%s remaining=%.3fs",
            self.session_id,
            self.turn_id,
            self.turn_revision,
            max(0, self.grace_deadline - asyncio.get_running_loop().time()),
        )

    async def _speech_started(self, event: SpeechStartedEvent) -> None:
        self.logger.debug(
            "Speaker speech start handling: session=%s state=%s turn=%s revision=%s reopened=%s current_turn=%s current_revision=%s",
            self.session_id,
            self.state,
            event.turn_id,
            event.turn_revision,
            event.reopened,
            self.turn_id,
            self.turn_revision,
        )
        if self.state == "grace" and event.reopened and self.turn_id == event.turn_id:
            if event.turn_revision != self.turn_revision + 1:
                await self._send_error("invalid_turn_revision", "Reopened turns must increment the revision by one.")
                return
            self._invalidate_pipeline()
            self.turn_revision = event.turn_revision
            self.state = "capturing"
            self.grace_held = False
            self.grace_changed.set()
            self.logger.info(
                "Speaker turn reopened: session=%s turn=%s revision=%s retained_bytes=%s",
                self.session_id,
                self.turn_id,
                self.turn_revision,
                len(self.audio),
            )
            return

        if self.state in {"grace", "responding"}:
            await self._cancel_response("barge_in")
        elif self.state == "capturing":
            await self._send_error("already_capturing", "A turn is already being captured.")
            return

        self.turn_id = event.turn_id
        self.turn_revision = event.turn_revision
        self.turn_language = self.input_language
        self.audio = bytearray()
        self.received_audio_frames = 0
        self.state = "capturing"

    async def _speech_soft_ended(
        self, event: SpeechSoftEndedEvent | InputLimitReachedEvent
    ) -> None:
        if self.state != "capturing" or not self._matches_turn(event):
            await self._send_error("invalid_soft_end", "Soft-end does not match the active turn.")
            return
        await self._start_generation()

    async def _start_generation(self) -> None:
        if not self.audio or self.turn_id is None:
            self.state = "idle"
            await self._send_json("input.transcription.empty")
            return
        self.response_generation += 1
        generation = self.response_generation
        self.current_generation = generation
        self.state = "grace"
        self.grace_deadline = (
            asyncio.get_running_loop().time() + self.reopen_grace_seconds
        )
        self.grace_held = False
        self.grace_changed.clear()
        self.playback_completed = asyncio.Event()
        self.audio_done = False
        self.segment_order = []
        self.segment_text = {}
        self.played_segments = set()
        self.first_audio_sent = False
        snapshot = bytes(self.audio)
        turn_id = self.turn_id
        revision = self.turn_revision
        language = self.turn_language or self.input_language
        self.logger.info(
            "Speaker soft-end: session=%s turn=%s revision=%s generation=%s bytes=%s grace_seconds=%.3f input_language=%s",
            self.session_id,
            turn_id,
            revision,
            generation,
            len(snapshot),
            self.reopen_grace_seconds,
            language,
        )
        self.pipeline_task = asyncio.create_task(
            self._run_pipeline(turn_id, revision, generation, snapshot, language)
        )

    async def _run_pipeline(
        self,
        turn_id: str,
        revision: int,
        generation: int,
        audio: bytes,
        language: SpeakerInputLanguage,
    ) -> None:
        text_task: asyncio.Task[None] | None = None
        stage_start = perf_counter()
        try:
            self.logger.info(
                "Speaker ASR start: session=%s turn=%s revision=%s generation=%s input_language=%s audio_seconds=%.3f",
                self.session_id,
                turn_id,
                revision,
                generation,
                language,
                len(audio) / (16_000 * 2),
            )
            transcript = (await self.services.transcribe(audio, language)).strip()
            self._require_current(turn_id, revision, generation)
            self.logger.info(
                "Speaker ASR complete: session=%s turn=%s revision=%s generation=%s input_language=%s elapsed=%.3fs chars=%s",
                self.session_id,
                turn_id,
                revision,
                generation,
                language,
                perf_counter() - stage_start,
                len(transcript),
            )
            if not transcript:
                await self._wait_for_grace(turn_id, revision, generation)
                self._require_current(turn_id, revision, generation)
                self.state = "idle"
                self.current_generation = None
                await self._send_json(
                    "input.transcription.empty",
                    turnId=turn_id,
                    turnRevision=revision,
                    responseGeneration=generation,
                )
                return

            assert self.context is not None
            speculative_history = [
                *self.context.history,
                {"role": "user", "content": transcript},
            ]
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            text_errors: list[BaseException] = []
            text_task = asyncio.create_task(
                self._produce_sentences(
                    turn_id,
                    revision,
                    generation,
                    speculative_history,
                    sentence_queue,
                    text_errors,
                )
            )

            await self._wait_for_grace(turn_id, revision, generation)
            self._require_current(turn_id, revision, generation)
            self.context.history.append({"role": "user", "content": transcript})
            self.state = "responding"
            await self._send_json(
                "input.transcription.committed",
                turnId=turn_id,
                turnRevision=revision,
                responseGeneration=generation,
                text=transcript,
            )
            await self._send_json(
                "response.started",
                turnId=turn_id,
                turnRevision=revision,
                responseGeneration=generation,
            )

            segment_index = 0
            while True:
                sentence = await sentence_queue.get()
                self._require_current(turn_id, revision, generation)
                if sentence is None:
                    break
                segment_index += 1
                segment_id = f"{generation}:{segment_index}"
                tts_start = perf_counter()
                self.logger.info(
                    "Speaker TTS start: session=%s turn=%s revision=%s generation=%s segment=%s chars=%s",
                    self.session_id,
                    turn_id,
                    revision,
                    generation,
                    segment_id,
                    len(sentence),
                )
                audio_bytes = await self.services.synthesize(
                    self.context.application,
                    sentence,
                    SpeakerSynthesisMetadata(
                        session_id=self.session_id,
                        turn_id=turn_id,
                        turn_revision=revision,
                        response_generation=generation,
                        segment_id=segment_id,
                    ),
                )
                self._require_current(turn_id, revision, generation)
                self.logger.info(
                    "Speaker TTS complete: session=%s turn=%s revision=%s generation=%s segment=%s elapsed=%.3fs bytes=%s",
                    self.session_id,
                    turn_id,
                    revision,
                    generation,
                    segment_id,
                    perf_counter() - tts_start,
                    len(audio_bytes),
                )
                self.segment_order.append(segment_id)
                self.segment_text[segment_id] = sentence
                await self._send_json(
                    "response.audio.segment_started",
                    turnId=turn_id,
                    turnRevision=revision,
                    responseGeneration=generation,
                    segmentId=segment_id,
                    text=sentence,
                    encoding="pcm_s16le",
                    sampleRate=24_000,
                )
                for offset in range(0, len(audio_bytes), OUTPUT_CHUNK_BYTES):
                    self._require_current(turn_id, revision, generation)
                    await self._send_bytes(audio_bytes[offset : offset + OUTPUT_CHUNK_BYTES])
                    if not self.first_audio_sent:
                        self.first_audio_sent = True
                        self.logger.info(
                            "Speaker first audio sent: session=%s turn=%s revision=%s generation=%s elapsed=%.3fs",
                            self.session_id,
                            turn_id,
                            revision,
                            generation,
                            perf_counter() - stage_start,
                        )
                await self._send_json(
                    "response.audio.segment_done",
                    responseGeneration=generation,
                    segmentId=segment_id,
                )

            if text_errors:
                raise text_errors[0]
            if not self.segment_order:
                raise RuntimeError("The language model produced no speakable text.")
            self.audio_done = True
            await self._send_json(
                "response.audio.done",
                responseGeneration=generation,
            )
            await self.playback_completed.wait()
        except asyncio.CancelledError:
            if text_task is not None and not text_task.done():
                text_task.cancel()
            raise
        except Exception as exc:
            if self._is_current(turn_id, revision, generation):
                self.logger.exception(
                    "Speaker pipeline failed: session=%s turn=%s revision=%s generation=%s",
                    self.session_id,
                    turn_id,
                    revision,
                    generation,
                )
                await self._send_error("pipeline_failed", str(exc))
                self.state = "idle"
                self.current_generation = None
        finally:
            if text_task is not None and not text_task.done():
                text_task.cancel()
            if text_task is not None:
                await asyncio.gather(text_task, return_exceptions=True)
            if self.pipeline_task is asyncio.current_task():
                self.pipeline_task = None

    async def _produce_sentences(
        self,
        turn_id: str,
        revision: int,
        generation: int,
        history: list[dict[str, str]],
        queue: asyncio.Queue[str | None],
        errors: list[BaseException],
    ) -> None:
        segmenter = SentenceSegmenter()
        first_token = True
        first_sentence = True
        try:
            assert self.context is not None
            async for token in self.services.stream_text(
                self.context.application,
                history,
            ):
                self._require_current(turn_id, revision, generation)
                if first_token:
                    first_token = False
                    self.logger.info(
                        "Speaker first LLM token: session=%s turn=%s revision=%s generation=%s",
                        self.session_id,
                        turn_id,
                        revision,
                        generation,
                    )
                for sentence in segmenter.feed(token):
                    if first_sentence:
                        first_sentence = False
                        self.logger.info(
                            "Speaker first complete sentence: session=%s turn=%s revision=%s generation=%s chars=%s",
                            self.session_id,
                            turn_id,
                            revision,
                            generation,
                            len(sentence),
                        )
                    self.logger.debug(
                        "Speaker sentence segmented: session=%s turn=%s revision=%s generation=%s chars=%s ellipsis=%s",
                        self.session_id,
                        turn_id,
                        revision,
                        generation,
                        len(sentence),
                        "..." in sentence or "…" in sentence,
                    )
                    await queue.put(sentence)
            for sentence in segmenter.finish():
                if first_sentence:
                    self.logger.info(
                        "Speaker first complete sentence at stream end: session=%s turn=%s revision=%s generation=%s chars=%s",
                        self.session_id,
                        turn_id,
                        revision,
                        generation,
                        len(sentence),
                    )
                await queue.put(sentence)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            errors.append(exc)
        finally:
            await queue.put(None)

    async def _wait_for_grace(
        self, turn_id: str, revision: int, generation: int
    ) -> None:
        while True:
            self._require_current(turn_id, revision, generation)
            remaining = self.grace_deadline - asyncio.get_running_loop().time()
            if remaining <= 0 and not self.grace_held:
                self.logger.info(
                    "Speaker grace opened: session=%s turn=%s revision=%s generation=%s",
                    self.session_id,
                    turn_id,
                    revision,
                    generation,
                )
                return
            self.logger.debug(
                "Speaker grace waiting: session=%s turn=%s revision=%s generation=%s remaining=%.3fs held=%s",
                self.session_id,
                turn_id,
                revision,
                generation,
                max(0, remaining),
                self.grace_held,
            )
            self.grace_changed.clear()
            try:
                if remaining > 0:
                    await asyncio.wait_for(self.grace_changed.wait(), timeout=remaining)
                else:
                    await self.grace_changed.wait()
            except TimeoutError:
                continue

    async def _segment_completed(
        self, event: PlaybackSegmentCompletedEvent
    ) -> None:
        if event.response_generation != self.current_generation:
            self.logger.debug(
                "Speaker stale segment acknowledgement dropped: session=%s current_generation=%s event_generation=%s segment=%s",
                self.session_id,
                self.current_generation,
                event.response_generation,
                event.segment_id,
            )
            return
        if event.segment_id in self.segment_text:
            self.played_segments.add(event.segment_id)
            self.logger.info(
                "Speaker segment played: session=%s generation=%s segment=%s",
                self.session_id,
                event.response_generation,
                event.segment_id,
            )

    async def _response_playback_completed(
        self, event: PlaybackResponseCompletedEvent
    ) -> None:
        if (
            self.state != "responding"
            or event.response_generation != self.current_generation
            or not self.audio_done
        ):
            return
        text = self._played_text()
        if text and self.context is not None:
            self.context.history.append({"role": "assistant", "content": text})
        await self._send_json(
            "response.completed",
            responseGeneration=event.response_generation,
            text=text,
        )
        self.state = "idle"
        self.current_generation = None
        self.playback_completed.set()

    async def _cancel_response(self, reason: str) -> None:
        generation = self.current_generation
        text = self._played_text()
        if text and self.context is not None:
            self.context.history.append({"role": "assistant", "content": text})
        self._invalidate_pipeline()
        self.logger.info(
            "Speaker response cancellation: session=%s generation=%s reason=%s acknowledged_segments=%s total_segments=%s",
            self.session_id,
            generation,
            reason,
            len(self.played_segments),
            len(self.segment_order),
        )
        if generation is not None:
            await self._send_json(
                "response.cancelled",
                responseGeneration=generation,
                reason=reason,
                text=text,
                finishReason="interrupted" if text else None,
            )
        self.current_generation = None
        self.state = "idle"

    def _played_text(self) -> str:
        return " ".join(
            self.segment_text[segment_id]
            for segment_id in self.segment_order
            if segment_id in self.played_segments
        ).strip()

    def _matches_turn(self, event: Any) -> bool:
        return event.turn_id == self.turn_id and event.turn_revision == self.turn_revision

    def _is_current(self, turn_id: str, revision: int, generation: int) -> bool:
        return (
            self.state != "closed"
            and self.turn_id == turn_id
            and self.turn_revision == revision
            and self.current_generation == generation
        )

    def _require_current(self, turn_id: str, revision: int, generation: int) -> None:
        if not self._is_current(turn_id, revision, generation):
            raise asyncio.CancelledError

    def _invalidate_pipeline(self) -> None:
        task = self.pipeline_task
        self.pipeline_task = None
        if task is not None and not task.done():
            self.logger.debug(
                "Speaker pipeline invalidated: session=%s task=%s",
                self.session_id,
                task.get_name(),
            )
            task.cancel()
        self.grace_changed.set()
        self.playback_completed.set()

    async def _send_json(self, event_type: str, **payload: Any) -> None:
        event = {
            "type": event_type,
            "eventId": str(uuid4()),
            "sessionId": self.session_id,
            **{key: value for key, value in payload.items() if value is not None},
        }
        async with self.send_lock:
            await self.websocket.send_json(event)

    async def _send_bytes(self, value: bytes) -> None:
        async with self.send_lock:
            await self.websocket.send_bytes(value)

    async def _send_error(self, code: str, message: str, fatal: bool = False) -> None:
        await self._send_json("error", code=code, message=message, fatal=fatal)
