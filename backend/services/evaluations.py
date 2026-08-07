from datetime import UTC, datetime
from collections.abc import Callable
from typing import Any, Literal

from mlflow.genai.judges import make_judge

from config import settings


EVALUATION_STAGES: list[tuple[str, int, str]] = [
    ("loading", 10, "Loading the saved conversation"),
    ("empathy", 25, "Assessing practitioner empathy"),
    ("professionalism", 40, "Assessing professionalism and boundaries"),
    ("relevance", 55, "Assessing relevance and context retention"),
    ("patient_experience", 70, "Assessing the mock patient's experience"),
    ("summary", 85, "Preparing learning feedback"),
    ("saving", 95, "Saving the evaluation"),
]

MetricName = Literal[
    "practitioner_empathy",
    "practitioner_professionalism",
    "practitioner_relevance",
    "mock_patient_frustration",
]


def select_evaluation_model(ollama_model: str) -> tuple[str, str, str | None]:
    """Return the judge model, provider label, and optional local base URL."""
    if settings.evaluation_model:
        return settings.evaluation_model, settings.evaluation_model.split(":/", 1)[0], None
    return f"ollama:/{ollama_model}", "ollama", settings.ollama_base_url


def normalize_conversation(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {"role": message["role"], "content": message["content"]}
        for message in value
        if isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]


def _create_judge(
    name: str,
    instructions: str,
    feedback_value_type: Any,
    model: str,
    base_url: str | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "name": name,
        "instructions": instructions,
        "feedback_value_type": feedback_value_type,
        "model": model,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return make_judge(**kwargs)


def _run_metric(
    name: MetricName,
    instructions: str,
    value_type: Any,
    conversation: list[dict[str, str]],
    model: str,
    base_url: str | None,
) -> dict[str, Any]:
    feedback = _create_judge(name, instructions, value_type, model, base_url)(
        inputs={"conversation": conversation}
    )
    value = getattr(feedback, "value", None)
    rationale = str(getattr(feedback, "rationale", "")).strip()
    if name == "mock_patient_frustration":
        level = str(value).lower()
        if level not in {"low", "moderate", "high"}:
            raise ValueError("The frustration judge returned an invalid level.")
        return {"level": level, "rationale": rationale}
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The {name} judge returned an invalid score.") from exc
    if score not in range(1, 6):
        raise ValueError(f"The {name} judge returned a score outside 1-5.")
    return {
        "score": score,
        "rationale": rationale,
        "improvement_suggestion": rationale,
    }


def evaluate_conversation(
    conversation: list[dict[str, str]],
    ollama_model: str,
    on_stage: Callable[[tuple[str, int, str]], None] | None = None,
) -> dict[str, Any]:
    model, provider, base_url = select_evaluation_model(ollama_model)
    base_instructions = """
You are evaluating a simulated therapist-training conversation. In {{ inputs }},
the role 'user' is the medical practitioner and the role 'assistant' is the mock
patient. This is educational feedback only: do not diagnose either party and do
not claim clinical certification. """
    results: dict[str, Any] = {
        "schema_version": 1,
        "judge_model": model,
        "provider": provider,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    results["practitioner_empathy"] = _run_metric(
        "practitioner_empathy",
        base_instructions
        + "Evaluate only user messages for empathy, kindness, validation, and rapport. "
        + "Return an integer score from 1 (poor) to 5 (excellent), then explain the score "
        + "and give one concrete improvement suggestion.",
        int,
        conversation,
        model,
        base_url,
    )
    if on_stage:
        on_stage(EVALUATION_STAGES[1])
    results["practitioner_professionalism"] = _run_metric(
        "practitioner_professionalism",
        base_instructions
        + "Evaluate only user messages for professionalism, respect, therapeutic boundaries, "
        + "and safe conduct. Return an integer score from 1 (poor) to 5 (excellent), then "
        + "explain the score and give one concrete improvement suggestion.",
        int,
        conversation,
        model,
        base_url,
    )
    if on_stage:
        on_stage(EVALUATION_STAGES[2])
    results["practitioner_relevance"] = _run_metric(
        "practitioner_relevance",
        base_instructions
        + "Evaluate the practitioner for responding to the mock patient's concerns, staying "
        + "on topic, and retaining relevant context. Return an integer score from 1 (poor) "
        + "to 5 (excellent), then explain the score and give one concrete improvement suggestion.",
        int,
        conversation,
        model,
        base_url,
    )
    if on_stage:
        on_stage(EVALUATION_STAGES[3])
    results["mock_patient_frustration"] = _run_metric(
        "mock_patient_frustration",
        base_instructions
        + "Assess mock-patient messages for frustration, confusion, or feeling unheard that "
        + "is plausibly caused or unresolved by the practitioner. Return exactly low, moderate, "
        + "or high, then explain the observed signals and how the practitioner could reduce them.",
        Literal["low", "moderate", "high"],
        conversation,
        model,
        base_url,
    )
    if on_stage:
        on_stage(EVALUATION_STAGES[4])
    feedback = _create_judge(
        "practitioner_learning_summary",
        base_instructions
        + "Give concise practical learning feedback: two strengths, two priorities for "
        + "improvement, and a short learning summary.",
        str,
        model,
        base_url,
    )(inputs={"conversation": conversation})
    summary = str(getattr(feedback, "rationale", "") or getattr(feedback, "value", "")).strip()
    results["overall_feedback"] = {
        "summary": summary,
        "strengths": "See learning summary.",
        "priorities": "See learning summary.",
    }
    if on_stage:
        on_stage(EVALUATION_STAGES[5])
    return results
