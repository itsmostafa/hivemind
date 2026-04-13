import json
import os
import sys
from datetime import datetime, timezone

import click
from coagent.advisor import Advisor
from coagent.config import load_config, merge_cli_overrides
from coagent.executor import ExecutorLoop
from coagent.log import NullTraceLogger, TraceLogger, configure_logging
from coagent.models import ModelClient
from coagent.policy import DecisionPolicy
from coagent.tracking import CostTracker


def _timestamped_trace_path(path: str) -> str:
    """Insert a UTC datetime stamp into a trace file path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if path.endswith("/") or path.endswith(os.sep) or os.path.isdir(path):
        return os.path.join(path.rstrip("/\\"), f"{ts}.jsonl")
    root, ext = os.path.splitext(path)
    return f"{root}_{ts}{ext or '.jsonl'}"


@click.group()
def cli() -> None:
    """Coagent: advisor strategy LLM framework."""


@cli.command()
@click.argument("task")
@click.option(
    "--executor", default=None, help="Executor model string (e.g. ollama/llama3)."
)
@click.option(
    "--executor-api-base",
    default=None,
    help="API base URL for executor (e.g. http://localhost:1234/v1).",
)
@click.option("--advisor", "advisor_model", default=None, help="Advisor model string.")
@click.option(
    "--advisor-api-base",
    default=None,
    help="API base URL for advisor (e.g. http://localhost:1234/v1).",
)
@click.option("--trace", default=None, help="Path to write JSONL trace file.")
@click.option(
    "--force-consult",
    is_flag=True,
    default=False,
    help="Force advisor consultation on the first policy check.",
)
def run(
    task: str,
    executor: str | None,
    executor_api_base: str | None,
    advisor_model: str | None,
    advisor_api_base: str | None,
    trace: str | None,
    force_consult: bool,
) -> None:
    """Run coagent on a TASK."""
    # Load and merge config (auto-discovers config.yaml / config.yml)
    config = load_config()
    config = merge_cli_overrides(
        config,
        executor=executor,
        advisor=advisor_model,
        executor_api_base=executor_api_base,
        advisor_api_base=advisor_api_base,
        force_consult=force_consult,
    )

    # Override trace file if provided via CLI
    if trace:
        config.logging.trace_file = trace

    # Set up logging
    configure_logging(config.logging.level)

    # Set up trace logger
    if config.logging.trace_file:
        trace_logger = TraceLogger(_timestamped_trace_path(config.logging.trace_file))
    else:
        trace_logger = NullTraceLogger()

    # Wire components
    executor_client = ModelClient(config.executor)
    advisor_client = ModelClient(config.advisor)
    advisor = Advisor(advisor_client)
    policy = DecisionPolicy(config.policy)
    tracker = CostTracker()

    # Run
    with trace_logger:
        loop = ExecutorLoop(
            executor_client=executor_client,
            advisor=advisor,
            policy=policy,
            tracker=tracker,
            trace_logger=trace_logger,
            config=config,
        )
        result = loop.run(task)

    # Output result
    click.echo("\n" + "=" * 60)
    click.echo("RESULT:")
    click.echo("=" * 60)
    click.echo(result.final_answer)
    click.echo("\n" + "=" * 60)
    click.echo("USAGE:")
    click.echo("=" * 60)
    _print_usage(result.usage_summary)

    if result.advisor_history:
        click.echo(f"\nAdvisor was consulted {len(result.advisor_history)} time(s).")

    sys.exit(0 if result.state.status == "completed" else 1)


def _print_usage(usage: dict) -> None:
    for role, data in usage.items():
        click.echo(
            f"  {role:10s}: {data['calls']} calls, "
            f"{data['prompt_tokens'] + data['completion_tokens']} tokens, "
            f"${data['cost_usd']:.2f}"
        )


@cli.command()
@click.argument("trace_file")
def trace(trace_file: str) -> None:
    """Pretty-print a JSONL trace file."""
    try:
        with open(trace_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("event", "unknown")
                    ts = event.get("ts", "")[:19]  # trim to seconds

                    if event_type == "executor_turn":
                        click.echo(
                            f"[{ts}] TURN {event.get('turn')} — executor ({event.get('tokens', {}).get('completion', '?')} tokens)"
                        )
                        content = event.get("content", "")
                        click.echo(
                            f"  {content[:200]}{'...' if len(content) > 200 else ''}"
                        )
                    elif event_type == "policy_check":
                        consult = event.get("should_consult")
                        reason = event.get("reason", "")
                        mark = "→ CONSULT" if consult else "  skip"
                        click.echo(
                            f"[{ts}] POLICY turn {event.get('turn')} {mark} ({reason})"
                        )
                    elif event_type == "advisor_call":
                        click.echo(
                            f"[{ts}] ADVISOR turn {event.get('turn')} — {event.get('advisor_status')} ({event.get('reason')})"
                        )
                        click.echo(f"  {event.get('advisor_diagnosis', '')[:200]}")
                    elif event_type == "run_complete":
                        usage = event.get("usage", {})
                        total = usage.get("total", {})
                        click.echo(
                            f"[{ts}] COMPLETE — {event.get('turns')} turns, {event.get('advisor_calls')} advisor calls, ${total.get('cost_usd', 0):.2f}"
                        )
                    else:
                        click.echo(f"[{ts}] {event_type}: {json.dumps(event)[:200]}")
                except json.JSONDecodeError:
                    click.echo(f"Line {i + 1}: (invalid JSON) {line[:100]}")
    except FileNotFoundError:
        click.echo(f"Error: trace file not found: {trace_file}", err=True)
        sys.exit(1)
