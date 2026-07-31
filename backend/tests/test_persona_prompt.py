import asyncio
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

fake_mlflow = types.ModuleType("mlflow")
fake_mlflow.get_tracking_uri = lambda: "test"
fake_mlflow.genai = types.SimpleNamespace()
sys.modules["mlflow"] = fake_mlflow

fake_soundfile = types.ModuleType("soundfile")
sys.modules["soundfile"] = fake_soundfile

fake_torch = types.ModuleType("torch")
fake_torch.Tensor = object
fake_torch.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    empty_cache=lambda: None,
)
fake_torch.bfloat16 = object()
fake_torch.float32 = object()
sys.modules["torch"] = fake_torch

fake_omnivoice = types.ModuleType("omnivoice")
fake_omnivoice.OmniVoice = type("OmniVoice", (), {})
fake_omnivoice.VoiceClonePrompt = type("VoiceClonePrompt", (), {})
sys.modules["omnivoice"] = fake_omnivoice

import app


class FakeResponse:
    def __init__(self, status_code: int, data: object):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse], payloads: list[dict]):
        self.responses = responses
        self.payloads = payloads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, _url: str, json: dict):
        self.payloads.append(json)
        return self.responses.pop(0)


class FakePrompt:
    template = "{{name}}|{{problem}}|{{background}}"
    variables = {"name", "problem", "background"}
    version = 7

    def format(self, **values: str):
        return f"{values['name']}|{values['problem']}|{values['background']}"


class PersonaExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_failure_adds_correction_before_retry(self):
        payloads: list[dict] = []
        client = FakeAsyncClient(
            [
                FakeResponse(
                    200,
                    {"message": {"content": '{"problem": 42}'}},
                ),
                FakeResponse(
                    200,
                    {
                        "message": {
                            "content": (
                                '{"problem":"Anxiety",'
                                '"background":"Avoids social situations."}'
                            )
                        }
                    },
                ),
            ],
            payloads,
        )

        with (
            patch.object(app.settings, "n_retries", 3),
            patch.object(app.httpx, "AsyncClient", return_value=client),
        ):
            profile = await app._extract_persona_profile("Persona instructions")

        self.assertEqual(profile.problem, "Anxiety")
        self.assertEqual(len(payloads), 2)
        retry_messages = payloads[1]["messages"]
        self.assertEqual(retry_messages[-2]["role"], "assistant")
        self.assertIn("Validation errors", retry_messages[-1]["content"])

    async def test_three_failed_attempts_raise_without_a_fourth(self):
        payloads: list[dict] = []
        client = FakeAsyncClient(
            [
                FakeResponse(200, {"message": {"content": "{}"}}),
                FakeResponse(200, {"message": {"content": "{}"}}),
                FakeResponse(200, {"message": {"content": "{}"}}),
            ],
            payloads,
        )

        with (
            patch.object(app.settings, "n_retries", 3),
            patch.object(app.httpx, "AsyncClient", return_value=client),
        ):
            with self.assertRaises(app.HTTPException) as raised:
                await app._extract_persona_profile("Persona instructions")

        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("3 attempts", raised.exception.detail)
        self.assertEqual(len(payloads), 3)

    async def test_transient_failure_uses_backoff_then_succeeds(self):
        payloads: list[dict] = []
        client = FakeAsyncClient(
            [
                FakeResponse(503, {}),
                FakeResponse(
                    200,
                    {
                        "message": {
                            "content": (
                                '{"problem":"Grief",'
                                '"background":"Recently lost a close friend."}'
                            )
                        }
                    },
                ),
            ],
            payloads,
        )

        with (
            patch.object(app.settings, "n_retries", 3),
            patch.object(app.httpx, "AsyncClient", return_value=client),
            patch.object(asyncio, "sleep", new=AsyncMock()) as sleep,
        ):
            profile = await app._extract_persona_profile("Persona instructions")

        self.assertEqual(profile.problem, "Grief")
        sleep.assert_awaited_once_with(0.25)
        self.assertEqual(len(payloads), 2)

    async def test_nonretryable_ollama_error_fails_immediately(self):
        payloads: list[dict] = []
        client = FakeAsyncClient([FakeResponse(400, {})], payloads)

        with (
            patch.object(app.settings, "n_retries", 3),
            patch.object(app.httpx, "AsyncClient", return_value=client),
        ):
            with self.assertRaises(app.HTTPException) as raised:
                await app._extract_persona_profile("Persona instructions")

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(len(payloads), 1)


class PersonaPromptCacheTests(unittest.IsolatedAsyncioTestCase):
    def test_instruction_change_produces_a_new_cache_key(self):
        prompt = FakePrompt()
        first = app.PersonaInput(
            name="Morgan",
            instruction_prompt="First instruction",
        )
        second = app.PersonaInput(
            name="Morgan",
            instruction_prompt="Changed instruction",
        )

        self.assertNotEqual(
            app._persona_cache_key(first, prompt),
            app._persona_cache_key(second, prompt),
        )

    async def test_cache_hit_skips_extraction(self):
        persona = app.PersonaInput(
            name="Morgan",
            instruction_prompt="Persona instructions",
        )
        prompt = FakePrompt()
        cache_key = app._persona_cache_key(persona, prompt)

        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "cache.sqlite3")
            with patch.object(app.settings, "persona_prompt_cache_path", cache_path):
                app._write_cached_system_prompt(cache_key, "cached")
                with (
                    patch.object(app, "load_prompt", return_value=prompt),
                    patch.object(
                        app,
                        "_extract_persona_profile",
                        new=AsyncMock(),
                    ) as extract,
                ):
                    result = await app._resolve_persona_system_prompt(
                        persona.name,
                        persona.instruction_prompt,
                    )

        self.assertEqual(result, "cached")
        extract.assert_not_awaited()

    async def test_cache_miss_extracts_renders_and_persists(self):
        persona = app.PersonaInput(
            name="Morgan",
            instruction_prompt="Persona instructions",
        )
        prompt = FakePrompt()
        profile = app.PersonaProfile(
            problem="Anxiety",
            background="Avoids unfamiliar social situations.",
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "cache.sqlite3")
            with (
                patch.object(app.settings, "persona_prompt_cache_path", cache_path),
                patch.object(app, "load_prompt", return_value=prompt),
                patch.object(
                    app,
                    "_extract_persona_profile",
                    new=AsyncMock(return_value=profile),
                ) as extract,
            ):
                result = await app._resolve_persona_system_prompt(
                    persona.name,
                    persona.instruction_prompt,
                )
                cached_result = await app._resolve_persona_system_prompt(
                    persona.name,
                    persona.instruction_prompt,
                )

        self.assertEqual(
            result,
            "Morgan|Anxiety|Avoids unfamiliar social situations.",
        )
        self.assertEqual(cached_result, result)
        extract.assert_awaited_once_with("Persona instructions")


class ConversationConstructionTests(unittest.TestCase):
    def test_backend_replaces_all_client_system_messages(self):
        request = app.InitiateRequest(
            persona_name="Morgan",
            instruction_prompt="Persona instructions",
            messages=[
                app.ChatMessage(role="system", content="untrusted"),
                app.ChatMessage(role="user", content="hello"),
            ],
        )

        result = app._stream_request_from_initiate_request(
            request,
            "trusted system prompt",
        )

        self.assertEqual(
            [message.model_dump() for message in result.messages],
            [
                {"role": "system", "content": "trusted system prompt"},
                {"role": "user", "content": "hello"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
