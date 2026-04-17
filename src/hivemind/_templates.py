DEFAULT_CONFIG_YAML = """\
# Executor model — runs the task end-to-end
executor:
  model: "ollama/llama3"
  api_base: "http://localhost:11434"
  # api_key not needed for local Ollama

# Advisor model — consulted only when executor needs help
advisor:
  model: "openai/gpt-4o"
  api_key: "${OPENAI_API_KEY}"   # env var reference

# Decision policy — when to consult the advisor
policy:
  max_advisor_calls: 5        # hard budget cap
  failure_threshold: 2        # consecutive failures before calling advisor
  confidence_threshold: 0.4   # executor self-reported confidence floor
  stagnation_turns: 4         # turns without progress before calling advisor
  cooldown_turns: 2           # min turns between advisor calls

max_turns: 20                 # max executor loop turns

logging:
  level: "INFO"               # DEBUG, INFO, WARNING
  # trace_file: "traces/run.jsonl"  # override via --trace flag; auto path used by default

# Web search tool — models decide when to call it (tool_choice="auto")
# Requires a Tavily API key: https://tavily.com
search:
  enabled: false              # set true or pass --search via CLI
  # api_key: "${TAVILY_API_KEY}"  # optional — TavilyClient reads TAVILY_API_KEY from env automatically
"""
