# Jev as a judge

A small sample of the evaluator in [Jev-as-a-Judge for Agent Evals](https://www.langchain.com/blog/jev-agent-evals-langsmith). The published run is [danielgshea/jev-as-a-judge](https://github.com/danielgshea/jev-as-a-judge). This folder keeps the judge and drops the weather agent, LangSmith, and the LLM comparison.

## What the post measured

Each frozen weather-agent run was scored twice:

| Signal | Meaning | How Jev produced it |
| --- | --- | --- |
| `quality` | Grounded, searched correctly, and useful | Mean of three yes/no probabilities, from 0 to 1 |
| `does_pass` | Overall success | A separate yes/no, recorded as 1 when the probability is at least 0.5 |

The repo also asks a third question, `outcome`: `answered`, `clarification_needed`, or `poor`.

Jev returns those values directly. An LLM judge generates text and a schema parser turns that text into a score. On five frozen traces repeated 100 times, Jev matched the human pass/fail label on all 500 decisions, and its quality scores barely moved (mean per-case variance `0.0000149`). Average cost in that run was about $0.00035 per call. That was one weather agent and one reviewer. A stable judge can still be stably wrong, so the pass/fail still needs a human label set before you trust it.

## What this sample changes

The experiment issued the quality checks, the pass/fail, and the outcome as separate calls. Those questions read the same trace and do not use each other's answers, so `judge_run` sends them in one `system_one` request. Code still owns the mean and the 0.5 cutoff. Changing the cutoff does not require another model call.

`quality` stays an average of three yes/no probabilities, which is what the variance table measured. A Score would be a different question: a position on a rubric you write, not the probability that a statement is true.

## Run

`uv sync` installs `jev_judge` and `usecases`. Put `OPENROUTER_API_KEY` in `.env` or the shell ([OpenRouter keys](https://openrouter.ai/keys)). Live commands call Jev at `https://openrouter.ai/api/v1/systemone` with model `~typesafe/jev-latest`. When a sample needs prose, it calls an OpenRouter chat model.

`uv run pytest` checks the score math and the harness wiring without calling the API.

### Judge

```bash
uv run python -m jev_judge
```

This scores the handwritten traces in `jev_judge/traces.py` and prints `agrees_with_sample_oracle` for `does_pass` and `outcome`.

| Trace | Oracle |
| --- | --- |
| `seattle-current` | pass, `answered` — forecast grounded in a search snippet |
| `springfield-ambiguous` | pass, `clarification_needed` — asks which Springfield, no search |
| `dublin-invented` | fail, `poor` — a sunny forecast with no search and no evidence |

`uv run pytest tests/test_judge.py` checks the mean of the three quality probabilities and the 0.5 pass cutoff.

LangSmith-shaped wrappers live on `langsmith_quality` and `langsmith_does_pass`. Each one calls Jev again. For a dataset run, call `judge_run` once and map that `Judgment` into both evaluator results.

### Use cases

The harness samples from [Building a Harness with Jev](https://www.youtube.com/watch?v=VE5dsWll06M) live in `usecases/`. They import the client and the chat models from `jev_judge`. Jev chooses; a chat model writes.

```bash
uv run python -m usecases
```

That runs every sample and prints a heading before each block of JSON.

| Heading | Module | What it does |
| --- | --- | --- |
| `model router` | `usecases/model_router.py` | Jev picks DeepSeek flash, GLM flash, or Nemotron Ultra. The chosen model writes a one- or two-sentence reply. Prompts: a capital lookup, a function rename, and a billing-service rollback plan. |
| `auto mode` | `usecases/auto_mode.py` | Jev scores the risk of a tool call. `JevAutoMode` blocks the call at risk ≥ 0.7 and does not run the tool. Calls: read `README.md`, `pytest -q`, and `rm -rf /`. |
| `supervisor choice` | `usecases/supervisor.py` | Jev picks a worker. Code turns that into `needs_writer`, a `task` handoff, or a relay of a worker report. It stops there and does not write the reply. Requests: the capital of France, a quarterly-letter comparison, and a parser fix. |
| `deep agent model slot` | `usecases/deep_stack.py` | Prints the error from `create_deep_agent(model="typesafe:jev-latest")`, then runs the parser request with `JevSupervisorModel` in that slot. The worker that writes is an OpenRouter chat model. |
| `email triage` | `usecases/email_triage.py` | Jev labels one message and scores how soon it needs a person. Code archives clear spam, files receipts and newsletters, and queues a reply. If the top label is close to the runner-up, a person picks. Messages: the Stripe connect failure from the harness post, a paid receipt, and a digest. |
| `trace feedback` | `usecases/trace_feedback.py` | One call returns the three online-eval keys from the LangSmith post: personal-data leakage, user intent, and frustration. Code alerts when leakage is at least 0.8 or frustration is at least 1.5. Traces: a card number copied into the reply, a how-to, and an angry outage report. |
| `relevant state` | `usecases/relevant_state.py` | Jev says, per field, whether that field helps answer the question. Code keeps fields at or above 0.5 and drops the rest before a later call. Case: a refund question, with the policy, the open order, and a restart log. |

`uv run pytest tests/test_usecases.py` checks the mapping from a Jev answer onto a model id, a block decision, a `task` call, and the Deep Agents model-string rejection.

What to look for in a live run:

| Sample | Check |
| --- | --- |
| `model_router` | Each line has `tier`, an OpenRouter `model` id, and a `reply` |
| `auto_mode` | `blocked` is false for the README read and for `pytest -q`. `blocked` is true for `rm -rf /` when `risk` is at least 0.7 |
| `supervisor` | The capital question is `needs_writer`. The quarterly-letter comparison delegates to `research`. The parser fix delegates to `code` |
| `deep_stack` | The first line rejects `typesafe:jev-latest`. The parser request then returns a `task` call and a `final` report from the worker |
| `email triage` | The Stripe failure is `queue` with a high `urgency`. The receipt and the digest are `file`. `human` appears when `margin` is below 0.2 |
| `trace feedback` | The card-number reply alerts `pii`. The how-to alerts nothing. The outage report alerts `frustration` |
| `relevant state` | `refund_policy` and `open_order` are in `kept`. `deploy_log` is not |

Jev is not the Deep Agents supervisor model. `create_deep_agent` accepts a chat model, and `typesafe:jev-latest` is rejected at graph build. `JevSupervisorModel` turns one Jev choice into a `task` call. The worker that writes is DeepSeek V4.1 Flash for a direct answer or drafting, GLM 5.3 Flash for code, and Nemotron 3 Ultra (`:free`) for research and architecture.
