# Coagent

Coagent implements the **advisor strategy** pattern: a cheap executor model handles tasks turn-by-turn, while a powerful advisor model is consulted only when the executor signals it needs help. The result is frontier-level performance at a fraction of the cost.

The typical setup pairs a local model as the executor with a state-of-the-art model (`claude-opus-4-6`, `gpt-4o`) as the advisor. The advisor is called sparingly, not on every turn.

Reference: https://claude.com/blog/the-advisor-strategy

## Quick Start

### Install

```bash
uv sync
```

### Run from CLI

```bash
# With Ollama (local)
coagent run --executor ollama/llama3.2 --advisor openai/gpt-5.4 "Explain REST vs GraphQL tradeoffs"

# With an OpenAI-compatible endpoint (e.g. LM Studio)
coagent run \
  --executor openai/local-model --executor-api-base http://localhost:1234/v1 \
  --advisor openai/gpt-4o \
  "Write a CSV parser in Python"

# With a config file (auto-discovered: config.yaml or config.yml in current directory)
coagent run "Write a CSV parser in Python"

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

Coagent automatically loads `config.yaml` or `config.yml` from the current directory if either exists. No flag required — just place the file and run.

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

# Optional: Tavily web search tool (models decide when to use it)
search:
  enabled: true
  # api_key is optional — TavilyClient reads TAVILY_API_KEY from env automatically
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
- OpenAI-compatible: set `api_base` in config, or pass `--executor-api-base` / `--advisor-api-base` via CLI

## Web Search

Enable Tavily web search so models can look up current information at their own discretion:

```bash
# Via CLI flag
TAVILY_API_KEY=tvly-xxx coagent run --search "What are the top AI papers this week?"

# Via config (search.enabled: true in config.yaml)
TAVILY_API_KEY=tvly-xxx coagent run "What are the top AI papers this week?"
```

The model decides when to call the search tool (`tool_choice="auto"`). It is never forced. Get a free API key at [tavily.com](https://tavily.com).

## Development

```bash
uv sync
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## Limitations (MVP)

- Synchronous only (no streaming or async)
- Single advisor model
