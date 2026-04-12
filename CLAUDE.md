# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all checks (lint + format + tests) — always run after implementing code changes
task check

# Individual tasks
task lint        # ruff check
task format      # ruff format
task test        # pytest

# Run a single test
uv run pytest tests/test_executor.py::test_name -v

# Invoke the CLI (use uv run — the system entrypoint may use the wrong Python)
uv run coagent run --executor ollama/llama3 --advisor ollama/llama3 "Your task"
uv run coagent trace traces/run.jsonl
```

## Architecture

Coagent implements the "advisor strategy" pattern: an **executor** model handles the task turn-by-turn while an **advisor** model is consulted only when needed, controlling cost.

### Data flow

```
User prompt → CLI / run_task() → ExecutorLoop
    → ModelClient (LiteLLM) → Executor model
    → DecisionPolicy.should_consult()  [checks gates + heuristics]
    → (if triggered) Advisor.consult() → Advisor model
    → inject AdvisorResponse as user message → next executor turn
    → [DONE] detected → return ExecutorResult
```

### Key components

| File | Role |
|---|---|
| `executor.py` | `ExecutorLoop` — drives conversation turns, injects advisor guidance, detects `[DONE]` |
| `policy.py` | `DecisionPolicy` — gates (budget/cooldown) + heuristics (explicit `[NEED_ADVICE]`, consecutive failures, low `[CONFIDENCE:X.X]`, stagnation via Jaccard similarity) |
| `advisor.py` | `Advisor` — serializes context to JSON, calls advisor model, parses structured `AdvisorResponse` |
| `models.py` | `ModelClient` — thin LiteLLM wrapper; passes `api_base` through for local endpoints |
| `schemas.py` | Pydantic models: `CoagentConfig`, `ModelConfig`, `PolicyConfig`, `ExecutorResult`, `AdvisorResponse` |
| `config.py` | `load_config()` + `merge_cli_overrides()` — YAML loading with `${ENV_VAR}` expansion |
| `log.py` | `TraceLogger` — structured JSONL events per turn |
| `tracking.py` | `CostTracker` — token/cost accounting per role (executor vs advisor) |
| `cli.py` | Click CLI — `coagent run` and `coagent trace` commands |

### Important details

- **LiteLLM routing**: `openai/*` models route to real OpenAI unless `api_base` is set. Always set `api_base` for local endpoints (LM Studio: `http://localhost:1234/v1`).
- **`ModelConfig.api_base`**: `str | None = None` — passed directly to LiteLLM.
- **CLI overrides**: `--executor-api-base` / `--advisor-api-base` flags feed into `merge_cli_overrides()`.
- **Advisor is never user-facing**: it only injects guidance as a user-turn message back to the executor.
- **Test fixtures**: `conftest.py` provides `sample_task`, `minimal_config`, `policy_config`; model calls are mocked.
