import asyncio
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

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

from app import app as fastapi_app
from routers import personas as persona_router
from schemas import (
    ChatMessage,
    InitiateRequest,
    OllamaModelsResponse,
    PersonaInput,
    PersonaPreparationRequest,
    PersonaProfile,
)
from services import ollama as ollama_service
from services import personas as persona_service
from services import requests as request_service
from state import persona_preparations


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


class FakeTagsResponse:
    def __init__(self, data: object):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeTagsClient:
    def __init__(self, result: FakeTagsResponse | Exception):
        self.result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, _url: str):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


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
            patch.object(persona_service.settings, "n_retries", 3),
            patch.object(persona_service.httpx, "AsyncClient", return_value=client),
        ):
            profile = await persona_service.extract_persona_profile("Persona instructions")

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
            patch.object(persona_service.settings, "n_retries", 3),
            patch.object(persona_service.httpx, "AsyncClient", return_value=client),
        ):
            with self.assertRaises(HTTPException) as raised:
                await persona_service.extract_persona_profile("Persona instructions")

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
            patch.object(persona_service.settings, "n_retries", 3),
            patch.object(persona_service.httpx, "AsyncClient", return_value=client),
            patch.object(asyncio, "sleep", new=AsyncMock()) as sleep,
        ):
            profile = await persona_service.extract_persona_profile("Persona instructions")

        self.assertEqual(profile.problem, "Grief")
        sleep.assert_awaited_once_with(0.25)
        self.assertEqual(len(payloads), 2)

    async def test_nonretryable_ollama_error_fails_immediately(self):
        payloads: list[dict] = []
        client = FakeAsyncClient([FakeResponse(400, {})], payloads)

        with (
            patch.object(persona_service.settings, "n_retries", 3),
            patch.object(persona_service.httpx, "AsyncClient", return_value=client),
        ):
            with self.assertRaises(HTTPException) as raised:
                await persona_service.extract_persona_profile("Persona instructions")

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(len(payloads), 1)


class PersonaPromptCacheTests(unittest.IsolatedAsyncioTestCase):
    def test_instruction_change_produces_a_new_cache_key(self):
        prompt = FakePrompt()
        first = PersonaInput(
            id="persona-1",
            name="Morgan",
            instruction_prompt="First instruction",
        )
        second = PersonaInput(
            id="persona-1",
            name="Morgan",
            instruction_prompt="Changed instruction",
        )

        self.assertNotEqual(
            persona_service.persona_cache_key(first, prompt),
            persona_service.persona_cache_key(second, prompt),
        )

    def test_duplicate_names_are_isolated_by_persona_id(self):
        prompt = FakePrompt()
        first = PersonaInput(
            id="persona-1",
            name="Morgan",
            instruction_prompt="Shared instructions",
        )
        second = PersonaInput(
            id="persona-2",
            name="Morgan",
            instruction_prompt="Shared instructions",
        )

        self.assertNotEqual(
            persona_service.persona_cache_key(first, prompt),
            persona_service.persona_cache_key(second, prompt),
        )

    def test_legacy_cache_schema_adds_persona_id(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "cache.sqlite3")
            connection = sqlite3.connect(cache_path)
            connection.execute(
                """
                CREATE TABLE persona_prompt_cache (
                    cache_key TEXT PRIMARY KEY,
                    persona_name TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.close()

            with patch.object(persona_service.settings, "persona_prompt_cache_path", cache_path):
                upgraded = persona_service.open_prompt_cache()
                columns = {
                    row[1]
                    for row in upgraded.execute(
                        "PRAGMA table_info(persona_prompt_cache)"
                    )
                }
                upgraded.close()

        self.assertIn("persona_id", columns)

    async def test_cache_hit_skips_extraction(self):
        persona = PersonaInput(
            id="persona-1",
            name="Morgan",
            instruction_prompt="Persona instructions",
        )
        prompt = FakePrompt()
        cache_key = persona_service.persona_cache_key(persona, prompt)

        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "cache.sqlite3")
            with patch.object(persona_service.settings, "persona_prompt_cache_path", cache_path):
                persona_service.write_cached_system_prompt(
                    cache_key,
                    persona.id,
                    persona.name,
                    "cached",
                )
                with (
                    patch.object(persona_service, "load_prompt", return_value=prompt),
                    patch.object(
                        persona_service,
                        "extract_persona_profile",
                        new=AsyncMock(),
                    ) as extract,
                ):
                    result = await persona_service.resolve_system_prompt(
                        persona.id,
                        persona.name,
                        persona.instruction_prompt,
                    )

        self.assertEqual(result, "cached")
        extract.assert_not_awaited()

    async def test_cache_miss_extracts_renders_and_persists(self):
        persona = PersonaInput(
            id="persona-1",
            name="Morgan",
            instruction_prompt="Persona instructions",
        )
        prompt = FakePrompt()
        profile = PersonaProfile(
            problem="Anxiety",
            background="Avoids unfamiliar social situations.",
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "cache.sqlite3")
            with (
                patch.object(persona_service.settings, "persona_prompt_cache_path", cache_path),
                patch.object(persona_service, "load_prompt", return_value=prompt),
                patch.object(
                    persona_service,
                    "extract_persona_profile",
                    new=AsyncMock(return_value=profile),
                ) as extract,
            ):
                result = await persona_service.resolve_system_prompt(
                    persona.id,
                    persona.name,
                    persona.instruction_prompt,
                )
                cached_result = await persona_service.resolve_system_prompt(
                    persona.id,
                    persona.name,
                    persona.instruction_prompt,
                )

        self.assertEqual(
            result,
            "Morgan|Anxiety|Avoids unfamiliar social situations.",
        )
        self.assertEqual(cached_result, result)
        extract.assert_awaited_once_with("Persona instructions")

    async def test_logs_created_prompt_and_changed_instruction_update(self):
        prompt = FakePrompt()
        profiles = [
            PersonaProfile(
                problem="Anxiety",
                background="Avoids unfamiliar social situations.",
            ),
            PersonaProfile(
                problem="Grief",
                background="Recently lost a close friend.",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            cache_path = str(Path(directory) / "cache.sqlite3")
            with (
                patch.object(persona_service.settings, "persona_prompt_cache_path", cache_path),
                patch.object(persona_service, "load_prompt", return_value=prompt),
                patch.object(
                    persona_service,
                    "extract_persona_profile",
                    new=AsyncMock(side_effect=profiles),
                ),
                self.assertLogs("uvicorn.error.p_gpt", level="INFO") as logs,
            ):
                await persona_service.resolve_system_prompt(
                    "persona-1",
                    "Morgan",
                    "First instruction",
                )
                await persona_service.resolve_system_prompt(
                    "persona-1",
                    "Morgan",
                    "Changed instruction",
                )

        output = "\n".join(logs.output)
        self.assertIn("Created persona system prompt", output)
        self.assertIn("Updated persona system prompt", output)
        self.assertIn("Morgan|Anxiety|Avoids unfamiliar social situations.", output)
        self.assertIn("Morgan|Grief|Recently lost a close friend.", output)


class PersonaPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        persona_preparations.clear()
        fastapi_app.state.voice_clone_prompts = {}
        fastapi_app.state.voice_clone_prompt_tasks = {}

    async def test_preparation_warms_changed_prompt_and_voice(self):
        request = PersonaPreparationRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Changed instructions",
            audio_sample_url="http://localhost:8090/api/files/personas/new.wav",
            prepare_system_prompt=True,
            prepare_voice_clone_prompt=True,
        )

        with (
            patch.object(
                request_service,
                "resolve_system_prompt",
                new=AsyncMock(return_value="system prompt"),
            ) as resolve,
            patch.object(
                request_service,
                "get_or_create_voice_clone_prompt",
                new=AsyncMock(return_value=object()),
            ) as prepare_voice,
        ):
            created = await persona_router.create_persona_preparation(
                request, types.SimpleNamespace(app=fastapi_app)
            )
            state = persona_preparations[created["id"]]
            self.assertIsNotNone(state.task)
            await state.task

        status = await persona_router.get_persona_preparation(created["id"])
        self.assertEqual(status["status"], "ready")
        resolve.assert_awaited_once_with(
            "persona-1", "Morgan", "Changed instructions"
        )
        prepare_voice.assert_awaited_once_with(
            fastapi_app, "http://localhost:8090/api/files/personas/new.wav"
        )

    async def test_voice_replacement_evicts_the_previous_cached_prompt(self):
        previous_url = "http://localhost:8090/api/files/personas/old.wav"
        fastapi_app.state.voice_clone_prompts[previous_url] = object()
        request = PersonaPreparationRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Instructions",
            audio_sample_url="http://localhost:8090/api/files/personas/new.wav",
            previous_audio_sample_url=previous_url,
            prepare_voice_clone_prompt=True,
        )

        with patch.object(
            request_service,
            "get_or_create_voice_clone_prompt",
            new=AsyncMock(return_value=object()),
        ):
            created = await persona_router.create_persona_preparation(
                request, types.SimpleNamespace(app=fastapi_app)
            )
            await persona_preparations[created["id"]].task

        self.assertNotIn(previous_url, fastapi_app.state.voice_clone_prompts)
        status = await persona_router.get_persona_preparation(created["id"])
        self.assertEqual(status["status"], "ready")

    async def test_failed_preparation_reports_an_error_without_discarding_state(self):
        request = PersonaPreparationRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Invalid instructions",
            prepare_system_prompt=True,
        )

        with patch.object(
            request_service,
            "resolve_system_prompt",
            new=AsyncMock(side_effect=RuntimeError("parser unavailable")),
        ):
            created = await persona_router.create_persona_preparation(
                request, types.SimpleNamespace(app=fastapi_app)
            )
            await persona_preparations[created["id"]].task

        status = await persona_router.get_persona_preparation(created["id"])
        self.assertEqual(status["status"], "error")
        self.assertEqual(status["error"], "parser unavailable")


class RequestSafetyTests(unittest.TestCase):
    def test_ui_numeric_values_are_unchanged_inside_ranges(self):
        request = InitiateRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Persona instructions",
            messages=[],
            temperature=1.25,
            max_tokens=512,
            repeat_penalty=1.1,
            seed=42,
            num_step=27,
        )

        self.assertEqual(request.temperature, 1.25)
        self.assertEqual(request.max_tokens, 512)
        self.assertEqual(request.repeat_penalty, 1.1)
        self.assertEqual(request.seed, 42)
        self.assertEqual(request.num_step, 27)

    def test_ui_numeric_values_are_clamped_to_frontend_ranges(self):
        low = InitiateRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Persona instructions",
            messages=[],
            temperature=-100,
            max_tokens=-100,
            repeat_penalty=-100,
            seed=-100,
            num_step=-100,
        )
        high = InitiateRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Persona instructions",
            messages=[],
            temperature=100,
            max_tokens=100_000,
            repeat_penalty=100,
            seed=10**30,
            num_step=100,
        )

        self.assertEqual(
            (low.temperature, low.max_tokens, low.repeat_penalty, low.seed, low.num_step),
            (0, 64, 1, 0, 22),
        )
        self.assertEqual(
            (
                high.temperature,
                high.max_tokens,
                high.repeat_penalty,
                high.seed,
                high.num_step,
            ),
            (2, 8192, 1.2, 9_007_199_254_740_991, 32),
        )

    def test_non_numeric_value_is_rejected(self):
        with self.assertRaises(ValidationError):
            InitiateRequest(
                persona_id="persona-1",
                persona_name="Morgan",
                instruction_prompt="Persona instructions",
                messages=[],
                temperature="malicious",
            )


class OllamaModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_listing_returns_installed_models(self):
        client = FakeTagsClient(
            FakeTagsResponse(
                {"models": [{"name": "configured"}, {"model": "second"}]}
            )
        )
        with (
            patch.object(ollama_service.settings, "ollama_text_model", "configured"),
            patch.object(ollama_service.httpx, "AsyncClient", return_value=client),
        ):
            result = await ollama_service.get_available_models()

        self.assertEqual(result.models, ["configured", "second"])
        self.assertFalse(result.used_fallback)

    async def test_model_listing_uses_configured_fallback_when_offline(self):
        client = FakeTagsClient(ollama_service.httpx.ConnectError("offline"))
        with (
            patch.object(ollama_service.settings, "ollama_text_model", "configured"),
            patch.object(ollama_service.httpx, "AsyncClient", return_value=client),
        ):
            result = await ollama_service.get_available_models()

        self.assertEqual(result.models, ["configured"])
        self.assertTrue(result.used_fallback)

    async def test_model_validation_accepts_reported_model(self):
        response = OllamaModelsResponse(
            models=["configured", "selected"],
            default_model="configured",
            used_fallback=False,
        )
        with patch.object(
            ollama_service,
            "get_available_models",
            new=AsyncMock(return_value=response),
        ):
            await ollama_service.validate_model("selected")

    async def test_model_validation_rejects_unavailable_model(self):
        response = OllamaModelsResponse(
            models=["configured"],
            default_model="configured",
            used_fallback=True,
        )
        with patch.object(
            ollama_service,
            "get_available_models",
            new=AsyncMock(return_value=response),
        ):
            with self.assertRaises(HTTPException) as raised:
                await ollama_service.validate_model("unavailable")

        self.assertEqual(raised.exception.status_code, 422)


class ConversationConstructionTests(unittest.TestCase):
    def test_backend_replaces_all_client_system_messages(self):
        request = InitiateRequest(
            persona_id="persona-1",
            persona_name="Morgan",
            instruction_prompt="Persona instructions",
            messages=[
                ChatMessage(role="system", content="untrusted"),
                ChatMessage(role="user", content="hello"),
            ],
        )

        result = request_service.stream_request_from_initiate_request(
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
