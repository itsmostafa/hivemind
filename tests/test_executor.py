"""Tests for ExecutorLoop in coagent.executor."""
from unittest.mock import MagicMock


from coagent.executor import ExecutorLoop
from coagent.log import NullTraceLogger
from coagent.policy import DecisionPolicy
from coagent.schemas import (
    AdvisorResponse,
    CoagentConfig,
    ModelConfig,
    ModelResponse,
    PolicyConfig,
)
from coagent.tracking import CostTracker


def make_mock_model_response(content: str) -> ModelResponse:
    """Helper to create a mock ModelResponse."""
    return ModelResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="test/model",
    )


def make_executor_loop(
    executor_responses: list[str],
    advisor_response: AdvisorResponse | None = None,
    max_turns: int = 5,
    policy_config: PolicyConfig | None = None,
) -> tuple[ExecutorLoop, MagicMock, MagicMock, CostTracker]:
    """Build an ExecutorLoop with mocked executor client and advisor."""
    if policy_config is None:
        policy_config = PolicyConfig(
            max_advisor_calls=3,
            failure_threshold=2,
            stagnation_turns=3,
            cooldown_turns=1,
        )

    config = CoagentConfig(
        executor=ModelConfig(model="test/executor"),
        advisor=ModelConfig(model="test/advisor"),
        policy=policy_config,
        max_turns=max_turns,
    )

    # Mock executor client
    mock_executor_client = MagicMock()
    mock_executor_client.generate.side_effect = [
        make_mock_model_response(content) for content in executor_responses
    ]

    # Mock advisor
    mock_advisor = MagicMock()
    if advisor_response is not None:
        mock_advisor.consult.return_value = (
            advisor_response,
            make_mock_model_response("advisor model output"),
        )

    tracker = CostTracker()
    policy = DecisionPolicy(policy_config)
    trace_logger = NullTraceLogger()

    loop = ExecutorLoop(
        executor_client=mock_executor_client,
        advisor=mock_advisor,
        policy=policy,
        tracker=tracker,
        trace_logger=trace_logger,
        config=config,
    )
    return loop, mock_executor_client, mock_advisor, tracker


def test_executor_completes_on_done():
    loop, _, _, _ = make_executor_loop(["Here is the final answer. [DONE]"])
    result = loop.run("What is 2+2?")

    assert result.state.status == "completed"
    # [DONE] should be stripped from the final answer
    assert "[DONE]" not in result.final_answer
    assert "final answer" in result.final_answer


def test_executor_max_turns():
    # Provide responses without [DONE] so loop exhausts max_turns.
    # Set max_advisor_calls=0 so the advisor budget gate always fires, preventing
    # stagnation/failure triggers from attempting to call the unmocked advisor.
    loop, _, _, _ = make_executor_loop(
        executor_responses=["Still working..."] * 3,
        max_turns=3,
        policy_config=PolicyConfig(max_advisor_calls=0),
    )
    result = loop.run("Solve this impossible task")

    assert result.state.status == "failed"
    assert result.state.turn_number == 3


def test_executor_consults_advisor_on_need_advice():
    advisor_resp = AdvisorResponse(
        status="continue",
        diagnosis="You are on the right track.",
        confidence=0.8,
    )
    # First response triggers advisor, second completes
    loop, _, mock_advisor, _ = make_executor_loop(
        executor_responses=["I need help. [NEED_ADVICE]", "Got it, done. [DONE]"],
        advisor_response=advisor_resp,
        max_turns=5,
    )
    result = loop.run("Do something hard")

    assert mock_advisor.consult.call_count == 1
    assert result.state.advisor_calls == 1
    assert result.state.status == "completed"


def test_executor_advisor_guidance_injected():
    advisor_resp = AdvisorResponse(
        status="revise",
        diagnosis="You are going in circles.",
        recommended_plan="Try a completely different approach.",
        confidence=0.9,
    )
    loop, _, _, _ = make_executor_loop(
        executor_responses=["[NEED_ADVICE]", "Revised approach. [DONE]"],
        advisor_response=advisor_resp,
        max_turns=5,
    )
    result = loop.run("Solve a hard problem")

    # Find the advisor guidance message injected into the conversation
    guidance_messages = [
        m for m in result.state.messages
        if m.get("role") == "user" and "[ADVISOR GUIDANCE" in m.get("content", "")
    ]
    assert len(guidance_messages) == 1
    guidance_content = guidance_messages[0]["content"]
    assert "revise" in guidance_content
    assert "You are going in circles." in guidance_content
    assert "Try a completely different approach." in guidance_content


def test_executor_usage_tracked():
    advisor_resp = AdvisorResponse(
        status="continue",
        diagnosis="Keep going.",
        confidence=0.7,
    )
    loop, _, _, tracker = make_executor_loop(
        executor_responses=["[NEED_ADVICE]", "Done. [DONE]"],
        advisor_response=advisor_resp,
        max_turns=5,
    )
    result = loop.run("Track my usage")

    summary = result.usage_summary
    # Executor was called twice (two turns)
    assert summary["executor"]["calls"] == 2
    # Advisor was called once
    assert summary["advisor"]["calls"] == 1
    # Total calls should sum correctly
    assert summary["total"]["calls"] == 3
    # Tokens and costs should be positive
    assert summary["executor"]["prompt_tokens"] > 0
    assert summary["advisor"]["cost_usd"] > 0
