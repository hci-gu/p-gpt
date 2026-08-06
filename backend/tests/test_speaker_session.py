import asyncio
import json
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speaker.protocol import parse_client_event
from speaker.asr import ParakeetASR
from speaker.session import (
    SentenceSegmenter,
    SpeakerConfiguredContext,
    SpeakerServices,
    SpeakerSession,
)


def event(event_type: str, **values):
    return parse_client_event(
        json.dumps({"type": event_type, "eventId": "event", **values})
    )


def configuration():
    return event(
        "session.configure",
        protocolVersion=1,
        personaId="persona-1",
        personaName="Morgan",
        instructionPrompt="Instructions",
        history=[{"role": "user", "content": "Earlier"}],
        generation={
            "model": "model",
            "temperature": 1,
            "repeatPenalty": 1,
            "maxTokens": 256,
            "cloneVoice": True,
            "numStep": 26,
        },
        inputAudio={
            "encoding": "pcm_s16le",
            "sampleRate": 16_000,
            "channels": 1,
            "frameSamples": 512,
        },
        outputAudio={
            "encoding": "pcm_s16le",
            "sampleRate": 24_000,
            "channels": 1,
        },
    )


class FakeWebSocket:
    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    async def send_json(self, value):
        self.sent.append(("json", value))

    async def send_bytes(self, value):
        self.sent.append(("bytes", value))

    async def close(self, code):
        self.sent.append(("close", code))


class FakeApplication:
    pass


def services(transcript="Hello", response="A first sentence. A second sentence."):
    async def configure(_event):
        return SpeakerConfiguredContext(
            application=FakeApplication(),
            history=[{"role": "system", "content": "trusted"}],
        )

    async def transcribe(_audio):
        return transcript

    async def stream_text(_context, _history):
        yield response

    async def synthesize(_context, _text):
        return bytes(4_800)

    return SpeakerServices(
        configure=configure,
        transcribe=transcribe,
        stream_text=stream_text,
        synthesize=synthesize,
    )


class SentenceSegmenterTests(unittest.TestCase):
    def test_complete_sentences_and_remainder(self):
        segmenter = SentenceSegmenter()
        self.assertEqual(segmenter.feed("Hello world. Still"), ["Hello world."])
        self.assertEqual(segmenter.feed(" here!"), [])
        self.assertEqual(segmenter.finish(), ["Still here!"])

    def test_long_unpunctuated_text_uses_word_boundary(self):
        segmenter = SentenceSegmenter(maximum_characters=20)
        result = segmenter.feed("one two three four five six")
        self.assertEqual(result, ["one two three four"])
        self.assertEqual(segmenter.finish(), ["five six"])

    def test_streamed_ellipsis_is_one_segment(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.feed("Wait."), [])
        self.assertEqual(segmenter.feed("."), [])
        self.assertEqual(segmenter.feed(". I need a moment"), ["Wait..."])
        self.assertEqual(segmenter.finish(), ["I need a moment"])

    def test_standalone_ellipsis_is_attached_to_following_text(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.feed("First. ... Second."), ["First."])
        self.assertEqual(segmenter.finish(), ["... Second."])

    def test_unicode_ellipsis_marks_a_pause(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.feed("Let me think… Then"), ["Let me think…"])
        self.assertEqual(segmenter.finish(), ["Then"])

    def test_punctuation_only_output_is_not_sent_to_tts(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.feed("..."), [])
        self.assertEqual(segmenter.finish(), [])


class ParakeetAdapterTests(unittest.TestCase):
    def test_pcm16_is_normalized_before_transcription(self):
        class FakeModel:
            def __init__(self):
                self.audio = None

            def transcribe(self, audio):
                self.audio = audio
                return " transcript "

        model = FakeModel()
        adapter = ParakeetASR(model=model, device="cpu")
        audio = np.asarray([-32_768, 0, 32_767], dtype="<i2").tobytes()

        self.assertEqual(adapter.transcribe_pcm16(audio), "transcript")
        np.testing.assert_allclose(model.audio, [-1, 0, 32_767 / 32_768])


class SpeakerSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = FakeWebSocket()
        self.session = SpeakerSession(
            self.websocket,
            services(),
            logging.getLogger("speaker-test"),
        )
        await self.session._configure(configuration())

    async def test_audio_is_rejected_outside_capture(self):
        await self.session._handle_audio(bytes(1_024))
        errors = [
            value
            for kind, value in self.websocket.sent
            if kind == "json" and value["type"] == "error"
        ]
        self.assertEqual(errors[-1]["code"], "audio_outside_capture")

    async def test_pipeline_commits_transcript_and_played_response(self):
        self.session.reopen_grace_seconds = 0
        with patch("speaker.session.REOPEN_GRACE_SECONDS", 0):
            await self.session._speech_started(
                event(
                    "input.speech_started",
                    turnId="turn-1",
                    turnRevision=0,
                    reopened=False,
                )
            )
            await self.session._handle_audio(bytes(1_024))
            await self.session._speech_soft_ended(
                event(
                    "input.speech_soft_ended",
                    turnId="turn-1",
                    turnRevision=0,
                )
            )

            for _ in range(50):
                await asyncio.sleep(0)
                if self.session.audio_done:
                    break

        self.assertTrue(self.session.audio_done)
        self.assertEqual(len(self.session.segment_order), 2)
        generation = self.session.current_generation
        self.assertIsNotNone(generation)
        for segment_id in self.session.segment_order:
            await self.session._segment_completed(
                event(
                    "playback.segment_completed",
                    responseGeneration=generation,
                    segmentId=segment_id,
                )
            )
        await self.session._response_playback_completed(
            event(
                "playback.response_completed",
                responseGeneration=generation,
            )
        )
        await asyncio.sleep(0)

        self.assertEqual(
            self.session.context.history,
            [
                {"role": "system", "content": "trusted"},
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "A first sentence. A second sentence.",
                },
            ],
        )

    async def test_reopen_increments_revision_and_preserves_audio(self):
        self.session.reopen_grace_seconds = 60
        with patch("speaker.session.REOPEN_GRACE_SECONDS", 60):
            await self.session._speech_started(
                event(
                    "input.speech_started",
                    turnId="turn-1",
                    turnRevision=0,
                    reopened=False,
                )
            )
            await self.session._handle_audio(bytes(1_024))
            await self.session._speech_soft_ended(
                event(
                    "input.speech_soft_ended",
                    turnId="turn-1",
                    turnRevision=0,
                )
            )
            previous_generation = self.session.current_generation
            await self.session._speech_started(
                event(
                    "input.speech_started",
                    turnId="turn-1",
                    turnRevision=1,
                    reopened=True,
                )
            )

        self.assertEqual(self.session.state, "capturing")
        self.assertEqual(self.session.turn_revision, 1)
        self.assertEqual(len(self.session.audio), 1_024)
        self.assertEqual(previous_generation, 1)
        self.assertIsNone(self.session.pipeline_task)

    async def test_interruption_keeps_only_acknowledged_segments(self):
        self.session.state = "responding"
        self.session.current_generation = 3
        self.session.segment_order = ["3:1", "3:2"]
        self.session.segment_text = {"3:1": "Played.", "3:2": "Not played."}
        self.session.played_segments = {"3:1"}

        await self.session._cancel_response("barge_in")

        self.assertEqual(
            self.session.context.history[-1],
            {"role": "assistant", "content": "Played."},
        )
        cancelled = [
            value
            for kind, value in self.websocket.sent
            if kind == "json" and value["type"] == "response.cancelled"
        ][-1]
        self.assertEqual(cancelled["text"], "Played.")
        self.assertEqual(cancelled["finishReason"], "interrupted")


if __name__ == "__main__":
    unittest.main()
