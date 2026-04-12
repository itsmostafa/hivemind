"""Tests for Pydantic model validation in coagent.schemas."""

import pytest
from pydantic import ValidationError

from coagent.config import load_config, merge_cli_overrides
from coagent.schemas import (
    AdvisorResponse,
    CoagentConfig,
    ExecutorResult,
    ExecutorState,
    ModelConfig,
    ModelResponse,
    PolicyConfig,
)


def test_model_response_valid():
    resp = ModelResponse(
        content="hello world",
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="test/model",
    )
    assert resp.content == "hello world"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5
    assert resp.cost == 0.001
    assert resp.model == "test/model"


def test_advisor_response_status_validation():
    with pytest.raises(ValidationError):
        AdvisorResponse(
            status="invalid_status",
            diagnosis="something went wrong",
            confidence=0.8,
        )


def test_policy_config_defaults():
    config = PolicyConfig()
    assert config.max_advisor_calls == 5
    assert config.failure_threshold == 2
    assert config.confidence_threshold == 0.4
    assert config.stagnation_turns == 4
    assert config.cooldown_turns == 2


def test_coagent_config_defaults():
    config = CoagentConfig(
        executor=ModelConfig(model="ollama/llama3"),
        advisor=ModelConfig(model="ollama/llama3"),
    )
    assert config.max_turns == 20
    assert isinstance(config.policy, PolicyConfig)
    assert config.policy.max_advisor_calls == 5
    assert config.logging.level == "INFO"
    assert config.logging.trace_file is None


def test_executor_state_defaults():
    state = ExecutorState(task="do something")
    assert state.turn_number == 0
    assert state.status == "running"
    assert state.advisor_calls == 0
    assert state.messages == []
    assert state.advisor_history == []


def test_executor_result_roundtrip():
    state = ExecutorState(task="compute 2+2", status="completed")
    result = ExecutorResult(
        final_answer="4",
        state=state,
        usage_summary={"executor": {}, "advisor": {}, "total": {}},
        advisor_history=[],
    )
    assert result.final_answer == "4"
    assert result.state.task == "compute 2+2"
    assert result.state.status == "completed"
    assert result.usage_summary["total"] == {}
    assert result.advisor_history == []


def test_merge_cli_overrides_sets_executor_api_base():
    config = load_config()
    result = merge_cli_overrides(
        config,
        executor="openai/gpt-oss-20b",
        executor_api_base="http://localhost:1234/v1",
    )
    assert result.executor.model == "openai/gpt-oss-20b"
    assert result.executor.api_base == "http://localhost:1234/v1"


def test_merge_cli_overrides_sets_advisor_api_base():
    config = load_config()
    result = merge_cli_overrides(
        config,
        advisor="openai/gpt-5.4-mini",
        advisor_api_base="http://localhost:1234/v1",
    )
    assert result.advisor.model == "openai/gpt-5.4-mini"
    assert result.advisor.api_base == "http://localhost:1234/v1"


def test_merge_cli_overrides_api_base_none_preserves_existing():
    config = load_config()
    # No api_base override → existing api_base (None) is preserved
    result = merge_cli_overrides(config, executor="openai/gpt-oss-20b")
    assert result.executor.api_base is None
