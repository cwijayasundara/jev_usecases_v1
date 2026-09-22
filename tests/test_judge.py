from jev_judge.judge import PASS_THRESHOLD, compose, judge_state
from jev_judge.traces import TRACES


def test_state_keeps_the_question_the_trace_and_the_expectation():
    trace = TRACES[0]
    state = judge_state(trace["inputs"], trace["outputs"], trace["reference_outputs"])
    assert state["user_question"] == "What is the weather in Seattle today?"
    assert state["expected_behavior"]["search_required"] is True
    assert state["tool_calls"] == ["tavily_search"]
    assert state["search_evidence"][0]["snippet"].startswith("Seattle:")


def test_quality_is_the_mean_of_the_three_checks():
    judgment = compose(
        {
            "is_grounded": 1.0,
            "matches_search_expectation": 0.5,
            "is_useful": 0.0,
            "does_pass": 0.2,
        },
        outcome="poor",
        outcome_confidence=0.8,
        outcome_probabilities={"answered": 0.1, "clarification_needed": 0.1, "poor": 0.8},
    )
    assert judgment.quality == 0.5
    assert judgment.does_pass is False
    assert judgment.does_pass_probability == 0.2


def test_pass_uses_the_experiment_threshold():
    below = compose(
        _nouls(PASS_THRESHOLD - 0.01),
        outcome="poor",
        outcome_confidence=1.0,
        outcome_probabilities={"poor": 1.0},
    )
    at = compose(
        _nouls(PASS_THRESHOLD),
        outcome="answered",
        outcome_confidence=1.0,
        outcome_probabilities={"answered": 1.0},
    )
    assert below.does_pass is False
    assert at.does_pass is True


def _nouls(does_pass: float) -> dict[str, float]:
    return {
        "is_grounded": 1.0,
        "matches_search_expectation": 1.0,
        "is_useful": 1.0,
        "does_pass": does_pass,
    }
