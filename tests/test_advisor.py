"""Tests for Advisor and build_advisor_context in coagent.advisor."""
import json
from unittest.mock import MagicMock


from coagent.advisor import Advisor, build_advisor_context
from coagent.schemas import AdvisorResponse, ExecutorState, ModelResponse


def make_mock_model_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="test/model",
    )


def make_advisor(response_content: str) -> tuple[Advisor, MagicMock]:
    """Return an Advisor whose ModelClient.generate returns the given content."""
    mock_client = MagicMock()
    mock_client.generate.return_value = make_mock_model_response(response_content)
    advisor = Advisor(client=mock_client)
    return advisor, mock_client


def make_valid_advisor_json(**overrides) -> str:
    data = {
        "status": "continue",
        "diagnosis": "You are on the right track.",
        "recommended_plan": None,
        "next_step": "Keep going.",
        "risks": [],
        "confidence": 0.85,
    }
    data.update(overrides)
    # Remove None values so Pydantic doesn't complain about Optional fields
    return json.dumps({k: v for k, v in data.items() if v is not None})


def make_advisor_context(state: ExecutorState, reason: str = "test_reason"):
    return build_advisor_context(state, reason)


def test_advisor_consult_success():
    json_content = make_valid_advisor_json(status="revise", diagnosis="Try a different approach.", confidence=0.7)
    advisor, _ = make_advisor(json_content)

    state = ExecutorState(task="some task")
    context = make_advisor_context(state)

    result, raw_response = advisor.consult(context)

    assert isinstance(result, AdvisorResponse)
    assert result.status == "revise"
    assert result.diagnosis == "Try a different approach."
    assert result.confidence == 0.7
    assert isinstance(raw_response, ModelResponse)


def test_advisor_consult_code_fence():
    json_body = make_valid_advisor_json(status="stop", diagnosis="Task is impossible.", confidence=0.9)
    fenced_content = f"```json\n{json_body}\n```"
    advisor, _ = make_advisor(fenced_content)

    state = ExecutorState(task="some task")
    context = make_advisor_context(state)

    result, _ = advisor.consult(context)

    assert isinstance(result, AdvisorResponse)
    assert result.status == "stop"
    assert result.diagnosis == "Task is impossible."


def test_advisor_consult_fallback():
    non_json_content = "Sorry, I cannot provide structured guidance right now."
    advisor, _ = make_advisor(non_json_content)

    state = ExecutorState(task="some task")
    context = make_advisor_context(state)

    result, _ = advisor.consult(context)

    assert isinstance(result, AdvisorResponse)
    assert result.status == "continue"
    assert result.diagnosis == non_json_content
    assert result.confidence == 0.5


def test_build_advisor_context_extracts_messages():
    state = ExecutorState(task="analyze data")
    # Add 5 assistant messages
    for i in range(5):
        state.messages.append({"role": "assistant", "content": f"Step {i+1} output"})

    context = build_advisor_context(state, reason="explicit_request", max_recent_turns=3)

    # Should have last 3 assistant messages joined by ---
    parts = context.recent_reasoning.split("\n---\n")
    assert len(parts) == 3
    assert parts[0] == "Step 3 output"
    assert parts[1] == "Step 4 output"
    assert parts[2] == "Step 5 output"
    assert context.current_blocker == "explicit_request"
    assert context.task == "analyze data"


def test_build_advisor_context_prior_calls():
    state = ExecutorState(task="plan a trip")
    # Add two prior advisor responses to history
    state.advisor_history.append(
        AdvisorResponse(status="continue", diagnosis="Good start.", confidence=0.8)
    )
    state.advisor_history.append(
        AdvisorResponse(status="revise", diagnosis="Change your approach.", confidence=0.6)
    )

    context = build_advisor_context(state, reason="stagnation")

    assert len(context.prior_advisor_calls) == 2
    assert context.prior_advisor_calls[0] == {"status": "continue", "diagnosis": "Good start."}
    assert context.prior_advisor_calls[1] == {"status": "revise", "diagnosis": "Change your approach."}
