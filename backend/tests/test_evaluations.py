from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import evaluations


class Feedback:
    def __init__(self, value: object, rationale: str):
        self.value = value
        self.rationale = rationale


class EvaluationServiceTests(unittest.TestCase):
    def test_normalize_conversation_keeps_supported_roles_and_text(self):
        conversation = evaluations.normalize_conversation(
            [
                {"role": "user", "content": "How have you been?"},
                {"role": "assistant", "content": "A little overwhelmed."},
                {"role": "tool", "content": "Ignore me"},
                {"role": "user", "content": "   "},
            ]
        )

        self.assertEqual(
            conversation,
            [
                {"role": "user", "content": "How have you been?"},
                {"role": "assistant", "content": "A little overwhelmed."},
            ],
        )

    def test_uses_ollama_model_without_a_cloud_override(self):
        with patch.object(evaluations.settings, "evaluation_model", None):
            model, provider, base_url = evaluations.select_evaluation_model("gemma4:e4b")

        self.assertEqual(model, "ollama:/gemma4:e4b")
        self.assertEqual(provider, "ollama")
        self.assertEqual(base_url, evaluations.settings.ollama_base_url)

    def test_cloud_override_wins_over_the_selected_ollama_model(self):
        with patch.object(evaluations.settings, "evaluation_model", "openai:/gpt-5-mini"):
            model, provider, base_url = evaluations.select_evaluation_model("gemma4:e4b")

        self.assertEqual(model, "openai:/gpt-5-mini")
        self.assertEqual(provider, "openai")
        self.assertIsNone(base_url)

    def test_evaluation_passes_the_complete_role_aware_conversation_to_each_judge(self):
        calls: list[dict] = []
        feedbacks = iter(
            [
                Feedback(4, "Warm acknowledgement. Improvement: ask a follow-up question."),
                Feedback(5, "Professional and respectful."),
                Feedback(3, "Mostly relevant."),
                Feedback("moderate", "The patient repeated their concern."),
                Feedback("summary", "Strengths: warmth. Priorities: clarify the concern."),
            ]
        )

        def make_judge(**kwargs):
            calls.append(kwargs)
            return lambda **_inputs: next(feedbacks)

        conversation = [
            {"role": "user", "content": "Tell me more about that."},
            {"role": "assistant", "content": "I still feel ignored."},
        ]
        with patch.object(evaluations, "make_judge", make_judge):
            result = evaluations.evaluate_conversation(conversation, "gemma4:e4b")

        self.assertEqual(result["practitioner_empathy"]["score"], 4)
        self.assertEqual(result["mock_patient_frustration"]["level"], "moderate")
        self.assertEqual(len(calls), 5)
        self.assertTrue(all("role 'user' is the medical practitioner" in call["instructions"] for call in calls))


if __name__ == "__main__":
    unittest.main()
