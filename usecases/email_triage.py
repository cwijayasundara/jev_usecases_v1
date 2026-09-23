"""Sort one inbox message before a person or a writer sees it.

Ryan Vogel's email triage is one of the production uses named at the end of
"Building a Harness with Jev". The Stripe ticket in that post is the urgent
sample. Jev returns a category, how the category probabilities are spread,
and an urgency score. Code owns the route.

The spread check follows the threshold discussion in the LangChain conversation:
the API confidence is a summary of how peaked the probability map is. When two
labels are close, the top label is a weak reason to auto-route, so the message
goes to a person. The cutoff is the gap between the top probability and the
second, which code can compute from the map.
"""

from dataclasses import dataclass

from typesafe_sdk import Choice, Score

from jev_judge.client import ask

CATEGORIES = {
    "needs_reply": "A person or a writer should answer this message.",
    "fyi": "A receipt, status, or record the reader may want to keep, with no question to answer.",
    "newsletter": "A mailing list, promotion, or digest nobody asked this inbox to answer.",
    "spam": "Unsolicited mail with no relationship to an account, order, or incident.",
}

# 0 can wait, 1 this week, 2 today. The score may fall between levels.
URGENCY_LEVELS = [
    "Nothing in the message is time-sensitive.",
    "It should be handled this week, and nothing is failing right now.",
    "Customers are blocked, money is being lost, or the sender asks for help now.",
]

# Top probability minus the runner-up. Below this, the labels are competing.
MARGIN_FLOOR = 0.2


@dataclass(frozen=True)
class Triage:
    category: str
    margin: float
    urgency: float
    action: str

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "margin": self.margin,
            "urgency": self.urgency,
            "action": self.action,
        }


def probability_margin(probabilities: dict[str, float]) -> float:
    ordered = sorted(probabilities.values(), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


def triage_from_answers(
    category: str,
    probabilities: dict[str, float],
    urgency: float,
    *,
    margin_floor: float = MARGIN_FLOOR,
) -> Triage:
    margin = probability_margin(probabilities)
    if margin < margin_floor:
        action = "human"
    elif category == "spam":
        action = "archive"
    elif category == "needs_reply":
        action = "queue"
    else:
        action = "file"
    return Triage(category=category, margin=margin, urgency=urgency, action=action)


def triage_message(subject: str, body: str) -> Triage:
    response = ask(
        {"subject": subject, "body": body},
        {
            "category": Choice(
                instructions="Which inbox category fits `subject` and `body`?",
                criteria=CATEGORIES,
            ),
            "urgency": Score(
                instructions="How soon does `subject` and `body` need a person?",
                criteria=URGENCY_LEVELS,
            ),
        },
    )
    category = response.choices["category"]
    return triage_from_answers(
        category.choice,
        dict(category.probabilities),
        response.scores["urgency"].score,
    )


MESSAGES = [
    {
        "subject": "Stripe connect keeps failing",
        "body": (
            "Hi, I've been trying to connect my Stripe account for 3 days and "
            "it keeps failing. I'm losing sales. Please help ASAP."
        ),
    },
    {
        "subject": "Your September receipt",
        "body": "Invoice 1842 for $49 was paid on September 2. No action is needed.",
    },
    {
        "subject": "Weekly product digest",
        "body": "This week's changelog and three articles from a list you subscribed to last year.",
    },
]
