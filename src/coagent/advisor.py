import json
import logging
from coagent.models import ModelClient
from coagent.schemas import AdvisorContext, AdvisorResponse, ExecutorState

logger = logging.getLogger(__name__)

ADVISOR_SYSTEM_PROMPT = """You are an advisor to an AI executor agent. Your role is to provide strategic guidance when the executor is stuck or needs help. You never produce final output for the user — only guidance for the executor.

You will receive context about the executor's current task and situation as JSON. Respond with a JSON object matching this exact schema:

{
  "status": "continue" | "stop" | "revise",
  "diagnosis": "what you think is happening (required)",
  "recommended_plan": "suggested plan or revision (optional)",
  "next_step": "immediate next action for the executor (optional)",
  "risks": ["list", "of", "risks"],
  "confidence": 0.0  // your confidence in this guidance, 0.0-1.0
}

Rules:
- status "continue": executor is on track, provide encouragement and tips
- status "revise": executor needs a different approach, provide recommended_plan
- status "stop": task is impossible or harmful, explain in diagnosis
- Always fill in diagnosis
- Respond with ONLY the JSON object, no other text
"""


class Advisor:
    """Consults the advisor model and parses its structured response."""

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def consult(self, context: AdvisorContext) -> AdvisorResponse:
        """Send context to advisor model and return parsed AdvisorResponse.

        On JSON parse failure, returns a fallback AdvisorResponse wrapping the raw text.
        """
        user_message = json.dumps(context.model_dump(), indent=2)
        messages = [{"role": "user", "content": user_message}]

        response = self.client.generate(messages, system=ADVISOR_SYSTEM_PROMPT)
        raw = response.content.strip()

        # Try to parse as JSON
        try:
            # Handle code blocks if model wraps in ```json ... ```
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])  # strip first and last line

            data = json.loads(raw)
            return AdvisorResponse.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Advisor response parse failed: %s. Using fallback.", e)
            return AdvisorResponse(
                status="continue",
                diagnosis=response.content,
                confidence=0.5,
            )


def build_advisor_context(
    state: ExecutorState,
    reason: str,
    max_recent_turns: int = 3,
) -> AdvisorContext:
    """Build an AdvisorContext from the current executor state.

    Extracts the last max_recent_turns assistant messages as recent_reasoning.
    Sets current_blocker to the reason the advisor was triggered.
    """
    # Collect recent assistant messages
    assistant_messages = [
        m["content"] for m in state.messages if m.get("role") == "assistant"
    ]
    recent = assistant_messages[-max_recent_turns:]
    recent_reasoning = "\n---\n".join(recent) if recent else "(no recent output)"

    # Summarize prior advisor calls
    prior_calls = [
        {"status": r.status, "diagnosis": r.diagnosis}
        for r in state.advisor_history
    ]

    return AdvisorContext(
        task=state.task,
        recent_reasoning=recent_reasoning,
        current_blocker=reason,
        prior_advisor_calls=prior_calls,
        turn_number=state.turn_number,
    )
