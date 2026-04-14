from unittest.mock import MagicMock

from coagent.events import (
    AdvisorCallEvent,
    PolicyCheckEvent,
    RunCompleteEvent,
    RunStartEvent,
    TurnCompleteEvent,
)
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


def test_turn_complete_event_fields():
    event = TurnCompleteEvent(
        turn=1,
        max_turns=20,
        content="Hello world",
        tool_calls=[],
        prompt_tokens=10,
        completion_tokens=5,
        cumulative_cost=0.001,
        status="running",
    )
    assert event.kind == "turn_complete"
    assert event.turn == 1
    assert event.cumulative_cost == 0.001


def test_run_start_event_fields():
    event = RunStartEvent(
        task="Do something",
        executor_model="ollama/llama3",
        advisor_model="ollama/llama3",
        max_turns=20,
    )
    assert event.kind == "run_start"
    assert event.task == "Do something"


def test_run_complete_event_fields():
    event = RunCompleteEvent(
        turns=5,
        advisor_calls=1,
        status="completed",
        final_answer="The answer is 42",
        usage={"total": {"cost_usd": 0.01}},
    )
    assert event.kind == "run_complete"
    assert event.final_answer == "The answer is 42"


def test_policy_check_event_fields():
    event = PolicyCheckEvent(
        turn=2,
        should_consult=True,
        reason="low_confidence",
    )
    assert event.kind == "policy_check"
    assert event.should_consult is True


def test_advisor_call_event_fields():
    event = AdvisorCallEvent(
        turn=2,
        reason="low_confidence",
        advisor_status="continue",
        diagnosis="You are on track.",
        confidence=0.8,
    )
    assert event.kind == "advisor_call"
    assert event.diagnosis == "You are on track."


def _mock_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="test/model",
        tool_calls=[],
    )


def _make_loop(
    responses: list[str],
    on_event=None,
    max_turns: int = 5,
    advisor_response: AdvisorResponse | None = None,
    policy_config: PolicyConfig | None = None,
):
    if policy_config is None:
        policy_config = PolicyConfig(max_advisor_calls=3, cooldown_turns=1)
    config = CoagentConfig(
        executor=ModelConfig(model="test/executor"),
        advisor=ModelConfig(model="test/advisor"),
        policy=policy_config,
        max_turns=max_turns,
    )
    mock_client = MagicMock()
    mock_client.generate.side_effect = [_mock_response(r) for r in responses]
    mock_client.model = "test/executor"

    mock_advisor = MagicMock()
    mock_advisor.client.model = "test/advisor"
    if advisor_response:
        mock_advisor.consult.return_value = (
            advisor_response,
            _mock_response("advisor output"),
        )

    return ExecutorLoop(
        executor_client=mock_client,
        advisor=mock_advisor,
        policy=DecisionPolicy(policy_config),
        tracker=CostTracker(),
        trace_logger=NullTraceLogger(),
        config=config,
        on_event=on_event,
    )


def test_no_callback_does_not_raise():
    loop = _make_loop(["Answer. [DONE]"], on_event=None)
    result = loop.run("test task")
    assert result.state.status == "completed"


def test_callback_receives_run_start():
    events = []
    loop = _make_loop(["Answer. [DONE]"], on_event=events.append)
    loop.run("test task")

    starts = [e for e in events if e.kind == "run_start"]
    assert len(starts) == 1
    assert starts[0].task == "test task"
    assert starts[0].executor_model == "test/executor"
    assert starts[0].advisor_model == "test/advisor"


def test_callback_receives_turn_complete():
    events = []
    loop = _make_loop(["Working...", "Done. [DONE]"], on_event=events.append)
    loop.run("task")

    turns = [e for e in events if e.kind == "turn_complete"]
    assert len(turns) == 2
    assert turns[0].turn == 1
    assert turns[1].turn == 2
    assert turns[1].cumulative_cost > 0


def test_callback_receives_policy_check():
    events = []
    loop = _make_loop(["Done. [DONE]"], on_event=events.append)
    loop.run("task")

    checks = [e for e in events if e.kind == "policy_check"]
    assert len(checks) == 1
    assert isinstance(checks[0].should_consult, bool)
    assert isinstance(checks[0].reason, str)


def test_callback_receives_advisor_call():
    events = []
    advisor_resp = AdvisorResponse(
        status="continue", diagnosis="Keep going.", confidence=0.8
    )
    loop = _make_loop(
        ["[NEED_ADVICE]", "Done. [DONE]"],
        on_event=events.append,
        advisor_response=advisor_resp,
    )
    loop.run("task")

    advisor_events = [e for e in events if e.kind == "advisor_call"]
    assert len(advisor_events) == 1
    assert advisor_events[0].diagnosis == "Keep going."
    assert advisor_events[0].advisor_status == "continue"


def test_callback_receives_run_complete():
    events = []
    loop = _make_loop(["Answer. [DONE]"], on_event=events.append)
    loop.run("task")

    completes = [e for e in events if e.kind == "run_complete"]
    assert len(completes) == 1
    assert completes[0].status == "completed"
    assert completes[0].final_answer == "Answer."


def test_event_order():
    events = []
    loop = _make_loop(["Done. [DONE]"], on_event=events.append)
    loop.run("task")

    kinds = [e.kind for e in events]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_complete"
    assert "turn_complete" in kinds
    assert "policy_check" in kinds
    assert kinds.index("turn_complete") < kinds.index("policy_check")
