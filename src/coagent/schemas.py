from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    model: str


class ModelConfig(BaseModel):
    model: str
    api_base: str | None = None
    api_key: str | None = None


class PolicyConfig(BaseModel):
    max_advisor_calls: int = 5
    failure_threshold: int = 2
    confidence_threshold: float = 0.4
    stagnation_turns: int = 4
    cooldown_turns: int = 2


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_file: str | None = None


class CoagentConfig(BaseModel):
    executor: ModelConfig
    advisor: ModelConfig
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    max_turns: int = 20
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class AdvisorContext(BaseModel):
    task: str
    current_plan: str | None = None
    recent_reasoning: str
    current_blocker: str | None = None
    prior_advisor_calls: list[dict] = Field(default_factory=list)
    turn_number: int
    executor_confidence: float | None = None


class AdvisorResponse(BaseModel):
    status: Literal["continue", "stop", "revise"]
    diagnosis: str
    recommended_plan: str | None = None
    next_step: str | None = None
    risks: list[str] = Field(default_factory=list)
    confidence: float


class ExecutorState(BaseModel):
    task: str
    messages: list[dict] = Field(default_factory=list)
    turn_number: int = 0
    advisor_calls: int = 0
    advisor_history: list[AdvisorResponse] = Field(default_factory=list)
    status: Literal["running", "completed", "failed"] = "running"


class UsageRecord(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class ExecutorResult(BaseModel):
    final_answer: str
    state: ExecutorState
    usage_summary: dict
    advisor_history: list[AdvisorResponse] = Field(default_factory=list)
