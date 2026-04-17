"""Tests for ExecutorLoop in hivemind.executor."""

from unittest.mock import MagicMock

from hivemind.executor import ExecutorLoop
from hivemind.log import NullTraceLogger
from hivemind.policy import DecisionPolicy
from hivemind.schemas import (
    AdvisorResponse,
    ExecutorState,
    HivemindConfig,
    ModelConfig,
    ModelResponse,
    PolicyConfig,
)
from hivemind.tracking import CostTracker


class RecordingTraceLogger:
    """Trace logger that records all events for test assertions."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def log(self, event: str, **data) -> None:
        self.events.append({"event": event, **data})

    def close(self) -> None:
        pass


def make_mock_model_response(
    content: str, tool_calls: list[dict] | None = None
) -> ModelResponse:
    """Helper to create a mock ModelResponse."""
    return ModelResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="test/model",
        tool_calls=tool_calls or [],
    )


def make_executor_loop(
    executor_responses: list[str],
    advisor_response: AdvisorResponse | None = None,
    max_turns: int = 5,
    policy_config: PolicyConfig | None = None,
    executor_model_responses: list[ModelResponse] | None = None,
    trace_logger=None,
) -> tuple[ExecutorLoop, MagicMock, MagicMock, CostTracker]:
    """Build an ExecutorLoop with mocked executor client and advisor."""
    if policy_config is None:
        policy_config = PolicyConfig(
            max_advisor_calls=3,
            failure_threshold=2,
            stagnation_turns=3,
            cooldown_turns=1,
        )

    config = HivemindConfig(
        executor=ModelConfig(model="test/executor"),
        advisor=ModelConfig(model="test/advisor"),
        policy=policy_config,
        max_turns=max_turns,
    )

    # Mock executor client
    mock_executor_client = MagicMock()
    if executor_model_responses is not None:
        mock_executor_client.generate.side_effect = executor_model_responses
    else:
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
    if trace_logger is None:
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
        m
        for m in result.state.messages
        if m.get("role") == "user" and "[ADVISOR GUIDANCE" in m.get("content", "")
    ]
    assert len(guidance_messages) == 1
    guidance_content = guidance_messages[0]["content"]
    assert "revise" in guidance_content
    assert "You are going in circles." in guidance_content
    assert "Try a completely different approach." in guidance_content


def test_failed_run_returns_last_non_empty_answer():
    # When the loop exhausts max_turns after real content followed by empty responses,
    # final_answer should be the last non-empty assistant response, not "".
    loop, _, _, _ = make_executor_loop(
        executor_responses=["Real answer here.", "", "", ""],
        max_turns=4,
        policy_config=PolicyConfig(max_advisor_calls=0),
    )
    result = loop.run("Answer my question")

    assert result.state.status == "failed"
    assert result.final_answer == "Real answer here."


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


def test_empty_content_injects_nudge_not_empty_assistant_message():
    # Empty response should become a user nudge, not an empty assistant message,
    # to prevent cascading empty turns.
    loop, _, _, _ = make_executor_loop(
        executor_responses=["", "Final answer. [DONE]"],
        policy_config=PolicyConfig(max_advisor_calls=0),
    )
    result = loop.run("Do something")

    assert result.state.status == "completed"
    # The empty turn should produce a user nudge, not an empty assistant message
    nudge_messages = [
        m
        for m in result.state.messages
        if m["role"] == "user" and "empty" in m["content"]
    ]
    assert len(nudge_messages) == 1
    empty_assistant_messages = [
        m
        for m in result.state.messages
        if m["role"] == "assistant" and not m["content"].strip()
    ]
    assert empty_assistant_messages == []


def test_run_accepts_existing_state_appends_user_message():
    loop, _, _, _ = make_executor_loop(
        executor_responses=["answer [DONE]"],
        policy_config=PolicyConfig(max_advisor_calls=0),
    )
    state = ExecutorState(task="original task")
    state.messages.append({"role": "user", "content": "original task"})
    state.messages.append({"role": "assistant", "content": "original answer"})

    result = loop.run("follow-up", state=state)

    assert result.state is state
    assert state.advisor_calls == 0
    # original user + original assistant + new user + new assistant
    assert len(state.messages) == 4
    assert state.messages[2] == {"role": "user", "content": "follow-up"}
    assert result.state.status == "completed"


def test_executor_logs_tool_calls_in_trace():
    """executor_turn trace event includes tool_calls; individual tool_call events are also logged."""
    recorder = RecordingTraceLogger()
    tool_calls = [{"tool": "tavily_search", "query": "latest AI news"}]
    responses = [
        make_mock_model_response(
            "Searched and found results. [DONE]", tool_calls=tool_calls
        ),
    ]
    loop, _, _, _ = make_executor_loop(
        executor_responses=[],
        executor_model_responses=responses,
        max_turns=5,
        policy_config=PolicyConfig(max_advisor_calls=0),
        trace_logger=recorder,
    )
    loop.run("Find latest AI news")

    # Individual tool_call event should be logged
    tool_call_events = [e for e in recorder.events if e["event"] == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["tool"] == "tavily_search"
    assert tool_call_events[0]["query"] == "latest AI news"

    # executor_turn event should include tool_calls
    turn_events = [e for e in recorder.events if e["event"] == "executor_turn"]
    assert len(turn_events) == 1
    assert turn_events[0]["tool_calls"] == tool_calls
