from usecases.auto_mode import RISK_THRESHOLD, verdict_from_risk
from usecases.deep_stack import _message_for, string_model_is_rejected
from usecases.model_router import route_from_choice
from usecases.supervisor import WorkerChoice, supervisor_action


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


def test_a_jev_model_string_is_not_a_deep_agent_model():
    message = string_model_is_rejected()
    assert "typesafe:jev-latest" not in message or "accepted" not in message
    assert "accepted typesafe:jev-latest" not in message
