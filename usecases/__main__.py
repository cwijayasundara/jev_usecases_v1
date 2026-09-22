"""Run the harness samples against Jev on OpenRouter."""

import json

from jev_judge.chat import complete
from jev_judge.client import load_local_env

from .auto_mode import CALLS, assess_tool_call
from .deep_stack import run_jev_supervisor, string_model_is_rejected
from .model_router import PROMPTS, route_request
from .supervisor import REQUESTS, choose_worker, supervisor_action


def main() -> None:
    load_local_env()

    print("\n== model router ==")
    for prompt in PROMPTS:
        route = route_request(prompt)
        reply = complete(route.model, prompt, system="Answer in one or two sentences.")
        print(json.dumps({"prompt": prompt, **route.as_dict(), "reply": reply}))

    print("\n== auto mode ==")
    for call in CALLS:
        verdict = assess_tool_call(call["user_request"], call["tool_name"], call["arguments"])
        print(json.dumps({"request": call["user_request"], "arguments": call["arguments"], **verdict.as_dict()}))

    print("\n== supervisor choice ==")
    for request in REQUESTS:
        choice = choose_worker(request)
        action = supervisor_action(choice, request)
        print(json.dumps({"request": request, "choice": choice.as_dict(), "action": action}))

    print("\n== deep agent model slot ==")
    print(string_model_is_rejected())
    code_request = REQUESTS[2]
    print(json.dumps({"request": code_request, **run_jev_supervisor(code_request)}))


if __name__ == "__main__":
    main()
