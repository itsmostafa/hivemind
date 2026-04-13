# Coagent: advisor strategy LLM framework
from coagent.config import load_config, merge_cli_overrides
from coagent.executor import ExecutorLoop
from coagent.schemas import CoagentConfig, ExecutorResult

__all__ = [
    "load_config",
    "merge_cli_overrides",
    "ExecutorLoop",
    "CoagentConfig",
    "ExecutorResult",
]


def run_task(task: str, config: CoagentConfig | None = None) -> ExecutorResult:
    """Convenience function to run a task with the given config."""
    from coagent.advisor import Advisor
    from coagent.log import NullTraceLogger
    from coagent.models import ModelClient
    from coagent.policy import DecisionPolicy
    from coagent.tracking import CostTracker

    if config is None:
        config = load_config()

    executor_client = ModelClient(config.executor, search=config.search)
    advisor_client = ModelClient(config.advisor, search=config.search)
    advisor = Advisor(advisor_client)
    policy = DecisionPolicy(config.policy)
    tracker = CostTracker()
    trace_logger = NullTraceLogger()

    loop = ExecutorLoop(
        executor_client=executor_client,
        advisor=advisor,
        policy=policy,
        tracker=tracker,
        trace_logger=trace_logger,
        config=config,
    )
    return loop.run(task)
