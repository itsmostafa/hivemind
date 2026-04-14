from coagent.events import (
    RunStartEvent,
    RunCompleteEvent,
    TurnCompleteEvent,
    PolicyCheckEvent,
    AdvisorCallEvent,
)


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
