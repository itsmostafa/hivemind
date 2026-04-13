from typing import Literal

from coagent.schemas import ModelResponse, UsageRecord


class CostTracker:
    """Tracks token usage and cost separately for executor and advisor."""

    def __init__(self) -> None:
        self._executor = UsageRecord()
        self._advisor = UsageRecord()

    def record(
        self, role: Literal["executor", "advisor"], response: ModelResponse
    ) -> None:
        """Record usage from a model response."""
        record = self._executor if role == "executor" else self._advisor
        record.calls += 1
        record.tool_calls += len(response.tool_calls)
        record.prompt_tokens += response.prompt_tokens
        record.completion_tokens += response.completion_tokens
        record.cost_usd += response.cost

    def summary(self) -> dict:
        """Return usage breakdown by role plus totals."""
        total = UsageRecord(
            calls=self._executor.calls + self._advisor.calls,
            tool_calls=self._executor.tool_calls + self._advisor.tool_calls,
            prompt_tokens=self._executor.prompt_tokens + self._advisor.prompt_tokens,
            completion_tokens=self._executor.completion_tokens
            + self._advisor.completion_tokens,
            cost_usd=self._executor.cost_usd + self._advisor.cost_usd,
        )
        return {
            "executor": self._executor.model_dump(),
            "advisor": self._advisor.model_dump(),
            "total": total.model_dump(),
        }
