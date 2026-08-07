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
from speaker.asr import KBWhisperASR, ParakeetASR, SpeakerASRRouter
from speaker.session import (
    REOPEN_GRACE_SECONDS,
    SentenceSegmenter,
    SpeakerConfiguredContext,
    SpeakerServices,
    SpeakerSession,
)
from speaker.tts_metrics import calculate_speaker_tts_timing


def event(event_type: str, **values):
    return parse_client_event(
        json.dumps({"type": event_type, "eventId": "event", **values})
    )


def configuration(input_language=None):
    language = {} if input_language is None else {"inputLanguage": input_language}
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
        **language,
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


def services(
    transcript="Hello",
    response="A first sentence. A second sentence.",
    transcribed_languages=None,
    synthesis_metadata=None,
):
    async def configure(_event):
        return SpeakerConfiguredContext(
            application=FakeApplication(),
            history=[{"role": "system", "content": "trusted"}],
        )

    async def transcribe(_audio, language):
        if transcribed_languages is not None:
            transcribed_languages.append(language)
        return transcript

    async def stream_text(_context, _history):
        yield response

    async def synthesize(_context, _text, metadata):
        if synthesis_metadata is not None:
            synthesis_metadata.append(metadata)
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


class SpeakerTtsTimingTests(unittest.TestCase):
    def test_lock_inference_encoding_total_and_real_time_are_distinct(self):
        timing = calculate_speaker_tts_timing(
            total_started_at=10,
            lock_requested_at=10.25,
            lock_acquired_at=10.75,
            inference_finished_at=12.75,
            encoding_started_at=12.8,
            completed_at=13,
            output_bytes=96_000,
            sample_rate=24_000,
        )

        self.assertAlmostEqual(timing.lock_wait_seconds, 0.5)
        self.assertAlmostEqual(timing.inference_seconds, 2.0)
        self.assertAlmostEqual(timing.encoding_seconds, 0.2)
        self.assertAlmostEqual(timing.total_seconds, 3.0)
        self.assertAlmostEqual(timing.audio_seconds, 2.0)
        self.assertAlmostEqual(timing.real_time_factor, 1.0)


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


class KBWhisperAdapterTests(unittest.TestCase):
    def test_pcm16_uses_swedish_chunked_transcription(self):
        calls = []

        def transcriber(audio, **options):
            calls.append((audio, options))
            return {"text": " hej världen "}

        adapter = KBWhisperASR(
            model=object(),
            processor=object(),
            transcriber=transcriber,
            device="cpu",
            dtype="float32",
            model_id="KBLab/kb-whisper-medium",
            revision="standard",
        )
        audio = np.asarray([-32_768, 0, 32_767], dtype="<i2").tobytes()

        self.assertEqual(adapter.transcribe_pcm16(audio), "hej världen")
        payload, options = calls[0]
        np.testing.assert_allclose(
            payload["raw"], [-1, 0, 32_767 / 32_768]
        )
        self.assertEqual(payload["sampling_rate"], 16_000)
        self.assertEqual(options["chunk_length_s"], 30)
        self.assertEqual(options["stride_length_s"], 5)
        self.assertEqual(
            options["generate_kwargs"],
            {"task": "transcribe", "language": "sv"},
        )

    def test_invalid_pipeline_result_is_rejected(self):
        adapter = KBWhisperASR(
            model=object(),
            processor=object(),
            transcriber=lambda *_args, **_kwargs: {},
            device="cpu",
            dtype="float32",
            model_id="KBLab/kb-whisper-medium",
            revision="standard",
        )

        with self.assertRaises(RuntimeError):
            adapter.transcribe_pcm16(bytes(2))

    def test_router_uses_selected_language(self):
        class FakeAdapter:
            def __init__(self, name):
                self.model_id = name
                self.device = "cpu"

            def transcribe_pcm16(self, _audio):
                return self.model_id

            def close(self):
                pass

        router = SpeakerASRRouter(
            routes={"en": FakeAdapter("parakeet"), "sv": FakeAdapter("whisper")}
        )

        self.assertEqual(router.transcribe_pcm16(bytes(2), "en"), "parakeet")
        self.assertEqual(router.transcribe_pcm16(bytes(2), "sv"), "whisper")


class SpeakerSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = FakeWebSocket()
        self.session = SpeakerSession(
            self.websocket,
            services(),
            logging.getLogger("speaker-test"),
        )
        await self.session._configure(configuration())

    async def test_configuration_defaults_to_english(self):
        self.assertEqual(REOPEN_GRACE_SECONDS, 2.0)
        self.assertEqual(self.session.input_language, "en")
        ready = [
            value
            for kind, value in self.websocket.sent
            if kind == "json" and value["type"] == "session.ready"
        ][-1]
        self.assertEqual(ready["inputLanguage"], "en")

    async def test_language_can_be_updated_while_idle(self):
        await self.session._update_session(
            event("session.update", inputLanguage="sv")
        )

        self.assertEqual(self.session.input_language, "sv")
        updated = [
            value
            for kind, value in self.websocket.sent
            if kind == "json" and value["type"] == "session.updated"
        ][-1]
        self.assertEqual(updated["inputLanguage"], "sv")

    async def test_language_update_is_rejected_during_active_turn(self):
        await self.session._speech_started(
            event(
                "input.speech_started",
                turnId="turn-1",
                turnRevision=0,
                reopened=False,
            )
        )
        await self.session._update_session(
            event("session.update", inputLanguage="sv")
        )

        self.assertEqual(self.session.input_language, "en")
        self.assertEqual(self.session.turn_language, "en")
        errors = [
            value
            for kind, value in self.websocket.sent
            if kind == "json" and value["type"] == "error"
        ]
        self.assertEqual(errors[-1]["code"], "session_busy")

    async def test_vad_diagnostics_accept_optional_recovery_and_timing_fields(self):
        diagnostic = event(
            "client.vad_diagnostic",
            activity="probability_summary",
            phase="idle",
            captureEpoch=4,
            detectionProfile="barge-in",
            pendingFrameCount=3,
            processingAverageMilliseconds=4.5,
            processingMaximumMilliseconds=7.5,
            queueDelayAverageMilliseconds=1.5,
            queueDelayMaximumMilliseconds=2.5,
            recoveryCount=1,
        )

        self.assertEqual(diagnostic.capture_epoch, 4)
        self.assertEqual(diagnostic.detection_profile, "barge-in")
        self.assertEqual(diagnostic.pending_frame_count, 3)
        self.assertEqual(diagnostic.processing_average_milliseconds, 4.5)

    async def test_candidate_hold_preserves_the_original_grace_deadline(self):
        loop = asyncio.get_running_loop()
        self.session.state = "grace"
        self.session.turn_id = "turn-hold"
        self.session.turn_revision = 0
        self.session.grace_deadline = loop.time() + 2
        original_deadline = self.session.grace_deadline

        await self.session._speech_candidate(
            event(
                "input.speech_candidate",
                turnId="turn-hold",
                turnRevision=0,
            )
        )
        self.assertTrue(self.session.grace_held)
        self.assertEqual(self.session.grace_deadline, original_deadline)

        await self.session._speech_candidate_cancelled(
            event(
                "input.speech_candidate_cancelled",
                turnId="turn-hold",
                turnRevision=0,
            )
        )
        self.assertFalse(self.session.grace_held)
        self.assertEqual(self.session.grace_deadline, original_deadline)

    async def test_synthesis_receives_turn_and_segment_metadata(self):
        metadata = []
        session = SpeakerSession(
            FakeWebSocket(),
            services(response="Hello.", synthesis_metadata=metadata),
            logging.getLogger("speaker-metadata-test"),
            reopen_grace_seconds=0,
        )
        await session._configure(configuration())
        await session._speech_started(
            event(
                "input.speech_started",
                turnId="turn-metadata",
                turnRevision=0,
                reopened=False,
            )
        )
        await session._handle_audio(bytes(1_024))
        await session._speech_soft_ended(
            event(
                "input.speech_soft_ended",
                turnId="turn-metadata",
                turnRevision=0,
            )
        )
        for _ in range(50):
            await asyncio.sleep(0)
            if metadata:
                break
        session._invalidate_pipeline()

        self.assertEqual(metadata[0].session_id, session.session_id)
        self.assertEqual(metadata[0].turn_id, "turn-metadata")
        self.assertEqual(metadata[0].turn_revision, 0)
        self.assertEqual(metadata[0].response_generation, 1)
        self.assertEqual(metadata[0].segment_id, "1:1")

    async def test_selected_language_is_passed_to_transcription(self):
        languages = []
        websocket = FakeWebSocket()
        session = SpeakerSession(
            websocket,
            services(transcribed_languages=languages),
            logging.getLogger("speaker-language-test"),
            reopen_grace_seconds=60,
        )
        await session._configure(configuration("sv"))
        await session._speech_started(
            event(
                "input.speech_started",
                turnId="turn-sv",
                turnRevision=0,
                reopened=False,
            )
        )
        await session._handle_audio(bytes(1_024))
        await session._speech_soft_ended(
            event(
                "input.speech_soft_ended",
                turnId="turn-sv",
                turnRevision=0,
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
            if languages:
                break
        session._invalidate_pipeline()

        self.assertEqual(languages, ["sv"])

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
        self.assertEqual(self.session.turn_language, "en")
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
