"""Drop state fields that do not help the next decision.

The end of the LangChain conversation is about context for Jev. A large state
full of fine-grained history made a later decision worse, so the useful step
is to keep the fields that bear on the question. Each field gets its own
yes/no in one request. Code keeps the fields whose probability is at least
the cutoff and leaves the rest out of the next call.
"""

from typesafe_sdk import Noul, NoulCriteria

from jev_judge.client import ask

RELEVANCE_CUTOFF = 0.5

_RELEVANCE = NoulCriteria(
    true="A person answering `question` would need this field.",
    false="This field does not change the answer to `question`.",
)


def kept_fields(relevance: dict[str, float], *, cutoff: float = RELEVANCE_CUTOFF) -> list[str]:
    return [name for name, probability in relevance.items() if probability >= cutoff]


def select_fields(question: str, fields: dict[str, str]) -> dict[str, float]:
    questions = {
        name: Noul(
            instructions=f"Does `fields.{name}` help answer `question`?",
            criteria=_RELEVANCE,
        )
        for name in fields
    }
    response = ask({"question": question, "fields": fields}, questions)
    return {name: response.nouls[name].noul for name in fields}


CASES = [
    {
        "question": "Can this customer get a refund for the duplicate charge?",
        "fields": {
            "refund_policy": "Duplicate charges are eligible for a refund.",
            "open_order": "Order A-104 was charged twice on September 2.",
            "deploy_log": "billing-api restarted 40 times between 02:01 and 02:04. Each line is one restart.",
        },
    }
]
