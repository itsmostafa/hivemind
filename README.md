# Coagent

An MVP Python framework for the "advisor strategy" LLM pattern: one executor model handles tasks end-to-end; an advisor model is consulted only when the executor needs help.

## Quick Start

### Install

```bash
pip install coagent
# or with uv:
uv add coagent
```

### Run from CLI

```bash
# With Ollama (local)
coagent run --executor ollama/llama3 --advisor ollama/llama3 "Explain REST vs GraphQL tradeoffs"

# With config file
coagent run --config config.yaml "Write a CSV parser in Python"

# View a trace
coagent trace traces/run.jsonl
```

### Python API

```python
from coagent import run_task, load_config
from coagent.schemas import CoagentConfig, ModelConfig

config = CoagentConfig(
    executor=ModelConfig(model="ollama/llama3", api_base="http://localhost:11434"),
    advisor=ModelConfig(model="openai/gpt-4o", api_key="..."),
)

result = run_task("Explain REST vs GraphQL tradeoffs", config=config)
print(result.final_answer)
print(result.usage_summary)
```

## Configuration

Copy `config.example.yaml` and edit for your setup:

```yaml
executor:
  model: "ollama/llama3"
  api_base: "http://localhost:11434"

advisor:
  model: "openai/gpt-4o"
  api_key: "${OPENAI_API_KEY}"

policy:
  max_advisor_calls: 5
  failure_threshold: 2
  confidence_threshold: 0.4
  stagnation_turns: 4
  cooldown_turns: 2

max_turns: 20
logging:
  level: "INFO"
  trace_file: "traces/run.jsonl"
```

## Architecture

```
User → CLI / Python API → ExecutorLoop
                              │
                    generate() via LiteLLM → Executor Model
                              │
                    DecisionPolicy.should_consult()
                              │
                    (if triggered) → Advisor Model
                              │
                    Parse AdvisorResponse (Pydantic)
                              │
                    Inject guidance → back to Executor
```

The executor is always in control. The advisor is a consulted resource — it never produces user-facing output.

## Advisor Triggers

The advisor is consulted when any of these fire:

| Trigger | Condition |
|---------|-----------|
| Explicit request | Executor outputs `[NEED_ADVICE]` |
| Consecutive failures | N turns with failure signals |
| Low confidence | Executor reports `[CONFIDENCE:0.3]` below threshold |
| Stagnation | Last N responses have high text overlap |

Gates prevent over-consulting: budget cap (`max_advisor_calls`) and cooldown (`cooldown_turns`).

## Supported Models

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

- Local: `ollama/llama3`, `ollama/mistral`
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-sonnet-4-6`
- OpenAI-compatible: set `api_base` to your endpoint

## Development

```bash
uv sync
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## Limitations (MVP)

- Text-only (no tool calling)
- Synchronous only (no streaming or async)
- Single advisor model
