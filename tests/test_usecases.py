from usecases.auto_mode import RISK_THRESHOLD, verdict_from_risk
from usecases.deep_stack import _message_for, string_model_is_rejected
from usecases.email_triage import MARGIN_FLOOR, triage_from_answers
from usecases.model_router import route_from_choice
from usecases.relevant_state import RELEVANCE_CUTOFF, kept_fields
from usecases.supervisor import WorkerChoice, supervisor_action
from usecases.trace_feedback import PII_ALERT, feedback_from_answers


def test_route_maps_the_tier_onto_a_chat_model():
    route = route_from_choice("glm", 0.9, {"deepseek": 0.1, "glm": 0.9, "nemotron": 0.0})
    assert route.tier == "glm"
    assert route.model == "z-ai/glm-5.3-flash"


def test_auto_mode_blocks_at_the_threshold():
    assert verdict_from_risk("execute", RISK_THRESHOLD).blocked is True
    assert verdict_from_risk("execute", RISK_THRESHOLD - 0.01).blocked is False


def test_supervisor_relays_a_worker_report_without_another_decision():
    choice = WorkerChoice("code", 1.0, {"code": 1.0})
    action = supervisor_action(choice, "fix the parser", tool_result="parser fixed")
    assert action == {"kind": "relay", "content": "parser fixed"}


def test_direct_answers_still_need_a_writer():
    choice = WorkerChoice("answer_directly", 0.8, {"answer_directly": 0.8})
    action = supervisor_action(choice, "What is 2 + 2?")
    assert action["kind"] == "needs_writer"


def test_a_worker_choice_becomes_a_task_call():
    action = supervisor_action(
        WorkerChoice("code", 0.95, {"code": 0.95}),
        "Fix the parser.",
    )
    message = _message_for(action)
    assert message.tool_calls[0]["name"] == "task"
    assert message.tool_calls[0]["args"]["subagent_type"] == "code"


def test_a_close_category_race_goes_to_a_person():
    triage = triage_from_answers(
        "needs_reply",
        {"needs_reply": 0.42, "fyi": 0.40, "newsletter": 0.10, "spam": 0.08},
        urgency=2.0,
    )
    assert triage.margin < MARGIN_FLOOR
    assert triage.action == "human"


def test_a_clear_spam_label_is_archived():
    triage = triage_from_answers(
        "spam",
        {"spam": 0.9, "newsletter": 0.1, "fyi": 0.0, "needs_reply": 0.0},
        urgency=0.1,
    )
    assert triage.action == "archive"


def test_a_clear_reply_is_queued_with_its_urgency():
    triage = triage_from_answers(
        "needs_reply",
        {"needs_reply": 0.8, "fyi": 0.2, "newsletter": 0.0, "spam": 0.0},
        urgency=1.8,
    )
    assert triage.action == "queue"
    assert triage.urgency == 1.8


def test_pii_and_frustration_raise_separate_alerts():
    leaked = feedback_from_answers(PII_ALERT, "billing", 0.9, frustration=0.2)
    angry = feedback_from_answers(0.01, "incident", 0.8, frustration=1.5)
    calm = feedback_from_answers(0.01, "how_to", 0.95, frustration=0.1)
    assert leaked.alerts == ("pii",)
    assert angry.alerts == ("frustration",)
    assert calm.alerts == ()


def test_fields_below_the_cutoff_are_left_out():
    kept = kept_fields(
        {"refund_policy": RELEVANCE_CUTOFF, "open_order": 0.9, "deploy_log": RELEVANCE_CUTOFF - 0.01}
    )
    assert kept == ["refund_policy", "open_order"]


def test_a_jev_model_string_is_not_a_deep_agent_model():
    message = string_model_is_rejected()
    assert "typesafe:jev-latest" not in message or "accepted" not in message
    assert "accepted typesafe:jev-latest" not in message
