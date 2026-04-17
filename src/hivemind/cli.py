import json
import os
import sys
from datetime import datetime

import click

from hivemind._templates import DEFAULT_CONFIG_YAML
from hivemind.advisor import Advisor
from hivemind import config as _config
from hivemind.config import load_config, merge_cli_overrides
from hivemind.executor import ExecutorLoop
from hivemind.log import NullTraceLogger, TraceLogger, configure_logging
from hivemind.models import ModelClient
from hivemind.policy import DecisionPolicy
from hivemind.tracking import CostTracker


def _timestamped_trace_path(path: str) -> str:
    """Return a trace file path using local datetime as the filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.endswith("/") or path.endswith(os.sep) or os.path.isdir(path):
        return os.path.join(path.rstrip("/\\"), f"{ts}.jsonl")
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    if ext:
        timestamped_name = f"{stem}_{ts}{ext}"
    else:
        timestamped_name = f"{filename}_{ts}.jsonl"
    return os.path.join(directory, timestamped_name) if directory else timestamped_name


@click.group()
def cli() -> None:
    """Hivemind: advisor strategy LLM framework."""


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init(force: bool) -> None:
    """Create ~/.hivemind/config.yml from the default template."""
    target = _config.USER_CONFIG_PATH
    if target.exists() and not force:
        click.echo(
            f"Error: config already exists at {target}. Use --force to overwrite.",
            err=True,
        )
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_CONFIG_YAML)
    click.echo(f"Created {target}")


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
@click.option(
    "--search",
    "search_enabled",
    is_flag=True,
    default=False,
    help="Enable Tavily web search tool.",
)
def run(
    task: str,
    executor: str | None,
    executor_api_base: str | None,
    advisor_model: str | None,
    advisor_api_base: str | None,
    trace: str | None,
    force_consult: bool,
    search_enabled: bool,
) -> None:
    """Run hivemind on a TASK."""
    # Load and merge config (auto-discovers config.yaml / config.yml)
    config = load_config()
    config = merge_cli_overrides(
        config,
        executor=executor,
        advisor=advisor_model,
        executor_api_base=executor_api_base,
        advisor_api_base=advisor_api_base,
        force_consult=force_consult,
        search_enabled=search_enabled or None,
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
    executor_client = ModelClient(config.executor, search=config.search)
    advisor_client = ModelClient(config.advisor, search=config.search)
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


@cli.command()
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
    "--search",
    "search_enabled",
    is_flag=True,
    default=False,
    help="Enable Tavily web search tool.",
)
def chat(
    executor: str | None,
    executor_api_base: str | None,
    advisor_model: str | None,
    advisor_api_base: str | None,
    trace: str | None,
    search_enabled: bool,
) -> None:
    """Start an interactive chat REPL."""
    config = load_config()
    config = merge_cli_overrides(
        config,
        executor=executor,
        advisor=advisor_model,
        executor_api_base=executor_api_base,
        advisor_api_base=advisor_api_base,
        search_enabled=search_enabled or None,
    )

    if trace:
        config.logging.trace_file = trace

    configure_logging(config.logging.level)

    if config.logging.trace_file:
        trace_logger = TraceLogger(_timestamped_trace_path(config.logging.trace_file))
    else:
        trace_logger = NullTraceLogger()

    executor_client = ModelClient(config.executor, search=config.search)
    advisor_client = ModelClient(config.advisor, search=config.search)
    advisor = Advisor(advisor_client)
    tracker = CostTracker()

    state = None

    with trace_logger:
        _print_chat_banner(
            config.executor.model, config.advisor.model, config.search.enabled
        )
        while True:
            try:
                user_input = click.prompt(
                    ">>", prompt_suffix=" ", default="", show_default=False
                )
            except (EOFError, KeyboardInterrupt, click.exceptions.Abort):
                break

            cmd = user_input.strip().lower()
            if not cmd:
                continue
            if cmd in ("q", "quit", "exit"):
                break
            if cmd in ("h", "help"):
                _print_chat_help()
                continue
            if cmd in ("r", "reset"):
                state = None
                tracker = CostTracker()
                trace_logger.log("chat_reset")
                click.echo("[session reset]")
                continue

            policy = DecisionPolicy(config.policy)
            loop = ExecutorLoop(
                executor_client, advisor, policy, tracker, trace_logger, config
            )
            result = loop.run(user_input.strip(), state=state)
            state = result.state
            click.echo(result.final_answer)

    _print_usage(tracker.summary())


def _print_usage(usage: dict) -> None:
    for role, data in usage.items():
        click.echo(
            f"  {role:10s}: {data['calls']} calls, {data['tool_calls']} tool calls, "
            f"{data['prompt_tokens'] + data['completion_tokens']} tokens, "
            f"${data['cost_usd']:.4f}"
        )


def _print_chat_banner(
    executor_model: str, advisor_model: str, search_enabled: bool
) -> None:
    search = "On" if search_enabled else "Off"
    click.echo("hivemind chat  (q=quit, r=reset, h=help)")
    click.echo(
        f"Executor Model: {executor_model}  Advisor Model: {advisor_model}  Web Search: {search}"
    )


def _print_chat_help() -> None:
    click.echo("  q / quit / exit  — end session")
    click.echo("  r / reset        — clear history and start fresh")
    click.echo("  h / help         — show this message")


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
                            f"[{ts}] COMPLETE — {event.get('turns')} turns, {event.get('advisor_calls')} advisor calls, ${total.get('cost_usd', 0):.4f}"
                        )
                    else:
                        click.echo(f"[{ts}] {event_type}: {json.dumps(event)[:200]}")
                except json.JSONDecodeError:
                    click.echo(f"Line {i + 1}: (invalid JSON) {line[:100]}")
    except FileNotFoundError:
        click.echo(f"Error: trace file not found: {trace_file}", err=True)
        sys.exit(1)
