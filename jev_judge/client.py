"""OpenRouter access to Jev."""

import os
from pathlib import Path

from typesafe_sdk import TypeSafeClient

# OpenRouter serves Jev on the TypeSafe System One API. Bare `jev-latest`
# is rewritten to this alias; passing the alias keeps the route explicit.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_MODEL = "~typesafe/jev-latest"


def load_local_env() -> None:
    """Fill missing variables from a project `.env`. Existing variables win."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    candidates = (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        if os.environ.get("OPENROUTER_API_KEY"):
            return


def openrouter_client() -> TypeSafeClient:
    load_local_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before calling Jev.")
    return TypeSafeClient(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
    )


def ask(state: object, questions: dict, *, model: str = OPENROUTER_MODEL):
    client = openrouter_client()
    try:
        return client.system_one(state=state, questions=questions, model=model)
    finally:
        client.close()
