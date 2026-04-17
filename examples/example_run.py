"""Example: running hivemind via the Python API."""

from hivemind import run_task
from hivemind.schemas import HivemindConfig, ModelConfig

# Option 1: Use a config file
# config = load_config("config.yaml")

# Option 2: Build config programmatically
config = HivemindConfig(
    executor=ModelConfig(model="ollama/llama3", api_base="http://localhost:11434"),
    advisor=ModelConfig(model="ollama/llama3", api_base="http://localhost:11434"),
)

task = "Explain the tradeoffs between REST and GraphQL APIs."

result = run_task(task, config=config)

print("=== RESULT ===")
print(result.final_answer)
print()
print("=== USAGE ===")
for role, data in result.usage_summary.items():
    print(
        f"  {role}: {data['calls']} calls, {data['prompt_tokens'] + data['completion_tokens']} tokens, ${data['cost_usd']:.6f}"
    )

if result.advisor_history:
    print(f"\nAdvisor was consulted {len(result.advisor_history)} time(s).")
    for i, resp in enumerate(result.advisor_history, 1):
        print(f"  Advisor {i}: status={resp.status}, diagnosis={resp.diagnosis[:100]}")
