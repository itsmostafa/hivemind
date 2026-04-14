from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RunStartEvent(BaseModel):
    kind: Literal["run_start"] = "run_start"
    task: str
    executor_model: str
    advisor_model: str
    max_turns: int


class TurnCompleteEvent(BaseModel):
    kind: Literal["turn_complete"] = "turn_complete"
    turn: int
    max_turns: int
    content: str
    tool_calls: list[dict] = Field(default_factory=list)
    prompt_tokens: int
    completion_tokens: int
    cumulative_cost: float
    status: str


class PolicyCheckEvent(BaseModel):
    kind: Literal["policy_check"] = "policy_check"
    turn: int
    should_consult: bool
    reason: str


class AdvisorCallEvent(BaseModel):
    kind: Literal["advisor_call"] = "advisor_call"
    turn: int
    reason: str
    advisor_status: str
    diagnosis: str
    recommended_plan: str | None = None
    next_step: str | None = None
    risks: list[str] = Field(default_factory=list)
    confidence: float


class RunCompleteEvent(BaseModel):
    kind: Literal["run_complete"] = "run_complete"
    turns: int
    advisor_calls: int
    status: str
    final_answer: str
    usage: dict


LoopEvent = (
    RunStartEvent
    | TurnCompleteEvent
    | PolicyCheckEvent
    | AdvisorCallEvent
    | RunCompleteEvent
)


@runtime_checkable
class EventCallback(Protocol):
    def __call__(self, event: LoopEvent) -> None: ...
