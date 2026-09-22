"""Pick the chat model for a turn.

This is the model-routing middleware from the harness walkthrough. Jev chooses
a named tier. Code maps that tier onto a tool-calling chat model. Jev is not
one of the tiers: it does not generate the reply.
"""

from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from typesafe_sdk import Choice

from jev_judge.chat import CHAT_MODELS
from jev_judge.client import ask

# OpenRouter model ids. Jev picks the tier; the named model writes the reply.
MODEL_CHOICES = CHAT_MODELS


@dataclass(frozen=True)
class ModelRoute:
    tier: str
    model: str
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "model": self.model,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def route_from_choice(choice: str, confidence: float, probabilities: dict[str, float]) -> ModelRoute:
    spec = MODEL_CHOICES[choice]
    return ModelRoute(
        tier=choice,
        model=spec["model"],
        confidence=confidence,
        probabilities=probabilities,
    )


def route_request(user_message: str) -> ModelRoute:
    response = ask(
        {"user_message": user_message},
        {
            "tier": Choice(
                instructions=(
                    "Which OpenRouter chat model should handle `user_message`? "
                    "Choose the smallest model whose criterion covers the task."
                ),
                criteria={name: spec["criteria"] for name, spec in MODEL_CHOICES.items()},
            )
        },
    )
    answer = response.choices["tier"]
    return route_from_choice(answer.choice, answer.confidence, dict(answer.probabilities))


class JevModelRouter(AgentMiddleware):
    """Swap the chat model before each Deep Agents model call.

    `model_factory` receives an OpenRouter model id and returns a chat model.
    """

    def __init__(self, model_factory=None):
        from jev_judge.chat import openrouter_chat

        self.model_factory = model_factory or openrouter_chat

    def _routed(self, request: ModelRequest) -> ModelRequest:
        route = route_request(_latest_user_text(request.messages))
        model: BaseChatModel = self.model_factory(route.model)
        return request.override(model=model)

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return handler(self._routed(request))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return await handler(self._routed(request))


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


PROMPTS = [
    "What is the capital of France?",
    "Rename the function `getUser` to `get_user` in src/accounts.py.",
    "Design a rollback plan for splitting the billing service out of the monolith.",
]
