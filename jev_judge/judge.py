"""Jev judge for one frozen agent run.

Same two signals as the LangSmith experiment, plus the outcome label from the
published repo:

- quality: mean of three yes/no probabilities (grounded, search behavior, useful)
- does_pass: a separate yes/no, turned into 0 or 1 at 0.5
- outcome: one of answered, clarification_needed, poor

The experiment sent those as separate calls. Questions over the same state do
not depend on each other, so this sample asks them together.
"""

from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafeClient

from .client import OPENROUTER_MODEL, openrouter_client

PASS_THRESHOLD = 0.5

QUALITY_IDS = ("is_grounded", "matches_search_expectation", "is_useful")

QUESTIONS = {
    "is_grounded": Noul(
        instructions=(
            "Does `final_answer` stay inside `search_evidence` and avoid "
            "inventing weather details?"
        ),
        criteria=NoulCriteria(
            true=(
                "Every weather claim is supported by `search_evidence`, or the "
                "answer makes no weather claim because it asks which place the user means."
            ),
            false="The answer states weather facts that `search_evidence` does not support.",
        ),
    ),
    "matches_search_expectation": Noul(
        instructions=(
            "Does `tool_calls` match `expected_behavior.search_required`?"
        ),
        criteria=NoulCriteria(
            true=(
                "A search tool is present when `expected_behavior.search_required` "
                "is true, and absent when it is false."
            ),
            false="The agent searched when it should have asked, or skipped a required search.",
        ),
    ),
    "is_useful": Noul(
        instructions="Is `final_answer` useful for `user_question`?",
        criteria=NoulCriteria(
            true=(
                "Names the place, addresses the requested time, and gives weather "
                "details, or asks which place was meant when the location is ambiguous."
            ),
            false=(
                "Omits the place or the requested time, withholds a required forecast, "
                "or answers an ambiguous place instead of asking."
            ),
        ),
    ),
    "does_pass": Noul(
        instructions="Does this run pass the expected weather-agent behavior?",
        criteria=NoulCriteria(
            true=(
                "If `expected_behavior.search_required` is true, the agent searched "
                "and answered from `search_evidence`. If it is false, the agent asked "
                "which place was meant and did not invent a forecast."
            ),
            false=(
                "The agent searched when it should have clarified, skipped a required "
                "search, invented weather details, or failed to address the question."
            ),
        ),
    ),
    "outcome": Choice(
        instructions="What outcome best describes `final_answer`?",
        criteria={
            "answered": (
                "A weather answer whose claims are supported by `search_evidence` "
                "and that covers the requested place and time."
            ),
            "clarification_needed": (
                "Asks which place the user means, and does not give a forecast "
                "for an ambiguous location."
            ),
            "poor": (
                "Misses the question, skips a required search, or states weather "
                "details that `search_evidence` does not support."
            ),
        },
    ),
}


@dataclass(frozen=True)
class Judgment:
    quality: float
    dimensions: dict[str, float]
    does_pass: bool
    does_pass_probability: float
    outcome: str
    outcome_confidence: float
    outcome_probabilities: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "quality": self.quality,
            "dimensions": self.dimensions,
            "does_pass": self.does_pass,
            "does_pass_probability": self.does_pass_probability,
            "outcome": self.outcome,
            "outcome_confidence": self.outcome_confidence,
            "outcome_probabilities": self.outcome_probabilities,
        }


def judge_state(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {
        "user_question": inputs["question"],
        "expected_behavior": reference_outputs,
        "final_answer": outputs["answer"],
        "tool_calls": outputs["tool_calls"],
        "search_evidence": outputs.get("evidence", []),
    }


def compose(
    nouls: dict[str, float],
    outcome: str,
    outcome_confidence: float,
    outcome_probabilities: dict[str, float],
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> Judgment:
    dimensions = {name: nouls[name] for name in QUALITY_IDS}
    probability = nouls["does_pass"]
    return Judgment(
        quality=sum(dimensions.values()) / len(dimensions),
        dimensions=dimensions,
        does_pass=probability >= pass_threshold,
        does_pass_probability=probability,
        outcome=outcome,
        outcome_confidence=outcome_confidence,
        outcome_probabilities=outcome_probabilities,
    )


def judge_run(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    *,
    client: TypeSafeClient | None = None,
    model: str = OPENROUTER_MODEL,
) -> Judgment:
    state = judge_state(inputs, outputs, reference_outputs)
    owns_client = client is None
    client = client or openrouter_client()
    try:
        response = client.system_one(state=state, questions=QUESTIONS, model=model)
    finally:
        if owns_client:
            client.close()

    outcome = response.choices["outcome"]
    return compose(
        {name: response.nouls[name].noul for name in (*QUALITY_IDS, "does_pass")},
        outcome.choice,
        outcome.confidence,
        dict(outcome.probabilities),
    )


def langsmith_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith evaluator shape: continuous quality in [0, 1]."""
    result = judge_run(inputs, outputs, reference_outputs)
    comment = ", ".join(f"{name}={value:.3f}" for name, value in result.dimensions.items())
    return {"key": "quality", "score": result.quality, "comment": comment}


def langsmith_does_pass(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith evaluator shape: binary pass, with the raw probability kept."""
    result = judge_run(inputs, outputs, reference_outputs)
    return {
        "key": "does_pass",
        "score": int(result.does_pass),
        "comment": f"probability={result.does_pass_probability:.3f}",
    }
