"""Chat models on OpenRouter. Jev stays on the decisions endpoint."""

from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter

from jev_judge.client import load_local_env

DEEPSEEK = "deepseek/deepseek-v4.1-flash"
GLM = "z-ai/glm-5.3-flash"
NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Ordered from the smallest task to the largest. Jev picks one; these models write.
CHAT_MODELS = {
    "deepseek": {
        "model": DEEPSEEK,
        "criteria": (
            "A direct lookup, a short factual answer, or a rename in a file "
            "the user already named."
        ),
    },
    "glm": {
        "model": GLM,
        "criteria": (
            "A code change that needs an implementation and a test, or a "
            "short multi-step edit in a repository."
        ),
    },
    "nemotron": {
        "model": NEMOTRON,
        "criteria": (
            "Architecture, a migration or rollback plan, or research that "
            "has to weigh several sources."
        ),
    },
}

# Worker name from the supervisor choice, then the chat model that writes.
WORKER_MODELS = {
    "research": NEMOTRON,
    "code": GLM,
    "writer": DEEPSEEK,
    "answer_directly": DEEPSEEK,
}


def openrouter_chat(model_id: str, *, max_tokens: int | None = None) -> ChatOpenRouter:
    load_local_env()
    # DeepSeek accepts reasoning disabled, which keeps the reply in `content`.
    # GLM rejects that setting, so it keeps a larger token budget instead.
    if model_id == DEEPSEEK:
        reasoning = {"effort": "none"}
        token_budget = max_tokens or 400
    else:
        reasoning = None
        token_budget = max_tokens or 1600
    return ChatOpenRouter(
        model=model_id,
        temperature=0,
        max_tokens=token_budget,
        reasoning=reasoning,
        timeout=120_000,
        max_retries=1,
    )


def message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or isinstance(block, dict)
        ]
        text = "".join(parts).strip()
        if text:
            return text
    return ""


def complete(model_id: str, user: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
    messages: list[tuple[str, str]] = []
    if system:
        messages.append(("system", system))
    messages.append(("user", user))
    result = openrouter_chat(model_id, max_tokens=max_tokens).invoke(messages)
    if isinstance(result, AIMessage):
        return message_text(result)
    return str(result)
