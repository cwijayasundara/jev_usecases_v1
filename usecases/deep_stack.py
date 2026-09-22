"""Put the Jev supervisor decision inside a Deep Agents graph.

`create_deep_agent(model=...)` requires a chat model. A provider string such as
`typesafe:jev-latest` is rejected, because Jev is not a chat provider. A thin
`BaseChatModel` can sit in that slot and turn one Jev choice into a `task`
tool call. When the worker returns, the same class copies the tool text out.
That copy is not a Jev generation.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from jev_judge.chat import WORKER_MODELS, complete
from .supervisor import WORKERS, choose_worker, supervisor_action

_DELEGATING_WORKERS = tuple(name for name in WORKERS if name != "answer_directly")


class OpenRouterWorker(BaseChatModel):
    """One OpenRouter completion for a delegated task.

    `bind_tools` returns this model unchanged so the subagent answers in text
    on the first call. The Deep Agents filesystem tools stay available to a
    normal chat model; this wrapper keeps the sample to a single generation.
    """

    model_id: str

    @property
    def _llm_type(self) -> str:
        return "openrouter-worker"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        request = _latest_human(messages)
        text = complete(
            self.model_id,
            request,
            system="Answer the task in one short paragraph.",
        )
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])


class JevSupervisorModel(BaseChatModel):
    """Chat-model adapter. The only model call it makes is a Jev worker choice."""

    @property
    def _llm_type(self) -> str:
        return "jev-supervisor"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_result = _latest_tool_text(messages)
        if tool_result is not None:
            action = {"kind": "relay", "content": tool_result}
        else:
            request = _latest_human(messages)
            action = supervisor_action(choose_worker(request), request)
            if action["kind"] == "needs_writer":
                action = {
                    "kind": "reply",
                    "content": complete(
                        WORKER_MODELS["answer_directly"],
                        request,
                        system="Answer in one or two sentences.",
                    ),
                }
        return ChatResult(generations=[ChatGeneration(message=_message_for(action))])


def _message_for(action: dict) -> AIMessage:
    if action["kind"] == "delegate":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "description": action["description"],
                        "subagent_type": action["subagent_type"],
                    },
                    "id": "jev-delegate",
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(content=action["content"])


def _latest_human(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def _latest_tool_text(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return None


def string_model_is_rejected() -> str:
    """A Jev model id is not a Deep Agents chat-model string."""
    from deepagents import create_deep_agent

    try:
        create_deep_agent(model="typesafe:jev-latest")
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return "create_deep_agent accepted typesafe:jev-latest"


def run_jev_supervisor(user_request: str) -> dict[str, Any]:
    """One Deep Agents run whose supervisor model is the Jev adapter."""
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=JevSupervisorModel(),
        system_prompt=(
            "You are a supervisor. Delegate research, code, and drafting to "
            "the matching subagent. Do not pretend to have written the report."
        ),
        subagents=[
            {
                "name": name,
                "description": description,
                "system_prompt": description,
                "model": OpenRouterWorker(model_id=WORKER_MODELS[name]),
            }
            for name, description in WORKERS.items()
            if name in _DELEGATING_WORKERS
        ],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_request}]},
        config={"recursion_limit": 12},
    )
    messages = result["messages"]
    final = messages[-1]
    tool_calls = [
        {"name": call["name"], "args": call["args"]}
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    ]
    content = final.content if isinstance(final.content, str) else str(final.content)
    return {"final": content, "tool_calls": tool_calls}
