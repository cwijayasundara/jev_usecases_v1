"""Choose which Deep Agents worker gets the turn.

The supervisor inside `create_deep_agent` is a chat model: it writes the `task`
tool call, then writes the reply the user sees after the worker returns. Jev
can make the first of those decisions. It cannot write the reply.

`supervisor_action` is the code that sits in that gap. A tool result is copied
through. A worker name becomes a `task` call. `answer_directly` means the turn
still needs a generative model.
"""

from dataclasses import dataclass

from typesafe_sdk import Choice

from jev_judge.client import ask

WORKERS = {
    "research": "Look up sources and return a cited summary. Does not edit files.",
    "code": "Change code, run tests, and report what changed.",
    "writer": "Draft or revise prose the user will read.",
    "answer_directly": (
        "A short factual reply the supervisor can give without another agent. "
        "No research, no file edits, no draft longer than a few sentences."
    ),
}


@dataclass(frozen=True)
class WorkerChoice:
    worker: str
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "worker": self.worker,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def choose_worker(user_request: str) -> WorkerChoice:
    response = ask(
        {
            "user_request": user_request,
            "workers": [
                {"name": name, "does": description} for name, description in WORKERS.items()
            ],
        },
        {
            "worker": Choice(
                instructions="Which worker should handle `user_request`?",
                criteria=WORKERS,
            )
        },
    )
    answer = response.choices["worker"]
    return WorkerChoice(answer.choice, answer.confidence, dict(answer.probabilities))


def supervisor_action(
    choice: WorkerChoice,
    user_request: str,
    *,
    tool_result: str | None = None,
) -> dict:
    """Turn a Jev choice into the next supervisor step.

    `tool_result` is set once a worker has already replied. Copying it is
    ordinary code. Jev is not asked again, and it is not asked to write prose.
    """
    if tool_result is not None:
        return {"kind": "relay", "content": tool_result}
    if choice.worker == "answer_directly":
        return {
            "kind": "needs_writer",
            "worker": choice.worker,
            "content": (
                "Jev routed this turn to answer_directly. A chat model still has "
                "to write the reply."
            ),
        }
    return {
        "kind": "delegate",
        "subagent_type": choice.worker,
        "description": user_request,
    }


REQUESTS = [
    "What is the capital of France?",
    "Compare the last three quarterly letters and list what changed in guidance.",
    "The parser drops the last character of each line. Fix it and add a test.",
]
