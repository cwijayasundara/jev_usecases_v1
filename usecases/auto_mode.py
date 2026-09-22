"""Block a tool call before it runs.

Auto mode in the walkthrough asks Jev whether a proposed call is safe, then
returns an error tool message instead of executing it. Only listed tools are
checked. The threshold lives in code.
"""

from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from typesafe_sdk import Noul, NoulCriteria

from jev_judge.client import ask

RISK_THRESHOLD = 0.7

RISK_QUESTION = Noul(
    instructions="Is this tool call risky to run without a person approving it?",
    criteria=NoulCriteria(
        true=(
            "It deletes or overwrites data, runs a destructive shell command, "
            "sends a message, spends money, or changes credentials or permissions."
        ),
        false=(
            "It only reads state, or it runs a test or a local command the user "
            "asked for that does not destroy data or grant access."
        ),
    ),
)


@dataclass(frozen=True)
class ToolVerdict:
    tool: str
    risk: float
    blocked: bool

    def as_dict(self) -> dict:
        return {"tool": self.tool, "risk": self.risk, "blocked": self.blocked}


def verdict_from_risk(tool: str, risk: float, *, threshold: float = RISK_THRESHOLD) -> ToolVerdict:
    return ToolVerdict(tool=tool, risk=risk, blocked=risk >= threshold)


def assess_tool_call(user_request: str, tool_name: str, arguments: dict) -> ToolVerdict:
    response = ask(
        {
            "user_request": user_request,
            "tool_name": tool_name,
            "arguments": arguments,
        },
        {"risky": RISK_QUESTION},
    )
    return verdict_from_risk(tool_name, response.nouls["risky"].noul)


class JevAutoMode(AgentMiddleware):
    """Refuse listed tool calls whose risk probability is at or above the threshold."""

    def __init__(self, tools: list[str], *, threshold: float = RISK_THRESHOLD):
        self.tools = set(tools)
        self.threshold = threshold

    def _gate(self, request):
        name = request.tool_call["name"]
        if name not in self.tools:
            return None
        verdict = assess_tool_call(
            _latest_user_text(request.state.get("messages", [])),
            name,
            request.tool_call.get("args") or {},
        )
        if not verdict.blocked:
            return None
        return ToolMessage(
            content=(
                f"Blocked before execution. Jev risk for `{name}` was "
                f"{verdict.risk:.2f}, threshold {self.threshold:.2f}."
            ),
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        blocked = self._gate(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        blocked = self._gate(request)
        return blocked if blocked is not None else await handler(request)


def _latest_user_text(messages) -> str:
    for message in reversed(messages):
        role = getattr(message, "type", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")
        if role not in ("human", "user"):
            continue
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        if isinstance(content, str):
            return content
    return ""


CALLS = [
    {
        "user_request": "Show me the top of the README.",
        "tool_name": "read_file",
        "arguments": {"path": "README.md"},
    },
    {
        "user_request": "Run the unit tests.",
        "tool_name": "execute",
        "arguments": {"command": "pytest -q"},
    },
    {
        "user_request": "Clean up the disk.",
        "tool_name": "execute",
        "arguments": {"command": "rm -rf /"},
    },
]
