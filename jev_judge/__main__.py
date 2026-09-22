import json
import os

from .client import load_local_env
from .judge import judge_run
from .traces import TRACES


def main() -> None:
    load_local_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "Set OPENROUTER_API_KEY (see .env.example) before scoring the sample traces."
        )

    for trace in TRACES:
        judgment = judge_run(
            trace["inputs"], trace["outputs"], trace["reference_outputs"]
        )
        oracle = trace["oracle"]
        print(
            json.dumps(
                {
                    "id": trace["id"],
                    "question": trace["inputs"]["question"],
                    "judgment": judgment.as_dict(),
                    "agrees_with_sample_oracle": {
                        "does_pass": judgment.does_pass == oracle["does_pass"],
                        "outcome": judgment.outcome == oracle["outcome"],
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
