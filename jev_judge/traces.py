"""Frozen weather-agent runs.

The LangSmith post scored captured traces, so the judge — not a live agent —
is the only moving part. These three stand in for that dataset. `oracle` is a
local label for this sample, not the human labels from the published experiment.
"""

TRACES = [
    {
        "id": "seattle-current",
        "oracle": {"does_pass": True, "outcome": "answered"},
        "inputs": {"question": "What is the weather in Seattle today?"},
        "reference_outputs": {"location": "Seattle", "search_required": True},
        "outputs": {
            "answer": (
                "Seattle is cloudy and 58°F this afternoon, with light rain "
                "expected this evening. Source: National Weather Service Seattle."
            ),
            "tool_calls": ["tavily_search"],
            "evidence": [
                {
                    "title": "NWS Seattle",
                    "url": "https://weather.gov/sew",
                    "snippet": "Seattle: Cloudy, 58°F. Light rain developing this evening.",
                }
            ],
        },
    },
    {
        "id": "springfield-ambiguous",
        "oracle": {"does_pass": True, "outcome": "clarification_needed"},
        "inputs": {"question": "What is the weather in Springfield today?"},
        "reference_outputs": {"location": "Springfield", "search_required": False},
        "outputs": {
            "answer": (
                "Several places are named Springfield. Which one do you mean — "
                "Illinois, Missouri, Massachusetts, or another?"
            ),
            "tool_calls": [],
            "evidence": [],
        },
    },
    {
        "id": "dublin-invented",
        "oracle": {"does_pass": False, "outcome": "poor"},
        "inputs": {"question": "Will I need an umbrella in Dublin tomorrow?"},
        "reference_outputs": {"location": "Dublin", "search_required": True},
        "outputs": {
            "answer": "No umbrella needed. Dublin will be sunny and 72°F all day tomorrow.",
            "tool_calls": [],
            "evidence": [],
        },
    },
]
