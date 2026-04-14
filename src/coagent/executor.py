import logging
from typing import Union

from coagent.advisor import Advisor, build_advisor_context
from coagent.events import (
    AdvisorCallEvent,
    EventCallback,
    LoopEvent,
    PolicyCheckEvent,
    RunCompleteEvent,
    RunStartEvent,
    TurnCompleteEvent,
)
from coagent.log import NullTraceLogger, TraceLogger
from coagent.models import ModelClient
from coagent.policy import DecisionPolicy
from coagent.schemas import CoagentConfig, ExecutorResult, ExecutorState
from coagent.tracking import CostTracker

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """You are an AI assistant working to complete a task given by the user. Work through the task step by step.

When you have fully completed the task and produced a final answer, end your response with exactly: [DONE]

If you are stuck, genuinely need strategic guidance, or the user explicitly asks you to consult the advisor, include exactly: [NEED_ADVICE]

You may optionally report your confidence in your current approach with: [CONFIDENCE:0.8] (replace 0.8 with a value from 0.0 to 1.0)

Be concise and focused. Do not pad your response unnecessarily."""


class ExecutorLoop:
    def __init__(
        self,
        executor_client: ModelClient,
        advisor: Advisor,
        policy: DecisionPolicy,
        tracker: CostTracker,
        trace_logger: Union[TraceLogger, NullTraceLogger],
        config: CoagentConfig,
        on_event: EventCallback | None = None,
    ) -> None:
        self.executor_client = executor_client
        self.advisor = advisor
        self.policy = policy
        self.tracker = tracker
        self.trace_logger = trace_logger
        self.config = config
        self._on_event = on_event

    def _emit(self, event: LoopEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                logger.exception("on_event callback raised; ignoring")

    def run(self, task: str) -> ExecutorResult:
        """Run the executor loop for a given task. Returns the final result."""
        state = ExecutorState(task=task)
        state.messages.append({"role": "user", "content": task})

        final_answer = ""

        self._emit(
            RunStartEvent(
                task=task,
                executor_model=self.executor_client.model,
                advisor_model=self.advisor.client.model,
                max_turns=self.config.max_turns,
            )
        )

        for turn in range(self.config.max_turns):
            state.turn_number = turn + 1
            logger.info("Executor turn %d", state.turn_number)

            # Generate executor response
            response = self.executor_client.generate(
                state.messages, system=EXECUTOR_SYSTEM_PROMPT
            )
            self.tracker.record("executor", response)
            content = response.content

            # Append to conversation
            state.messages.append({"role": "assistant", "content": content})

            # Log any tool calls that occurred during generation
            for tc in response.tool_calls:
                self.trace_logger.log(
                    "tool_call",
                    turn=state.turn_number,
                    **tc,
                )

            # Log the turn
            self.trace_logger.log(
                "executor_turn",
                turn=state.turn_number,
                content=content[:500],  # truncate for trace
                tool_calls=response.tool_calls,
                tokens={
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                },
            )

            self._emit(
                TurnCompleteEvent(
                    turn=state.turn_number,
                    max_turns=self.config.max_turns,
                    content=content,
                    tool_calls=response.tool_calls,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    cumulative_cost=self.tracker.summary()["total"]["cost_usd"],
                    status=state.status,
                )
            )

            # Evaluate policy (before done check so force_consult fires even on
            # turns where the executor signals completion)
            should_consult, reason = self.policy.should_consult(state, content)

            self.trace_logger.log(
                "policy_check",
                turn=state.turn_number,
                should_consult=should_consult,
                reason=reason,
            )

            self._emit(
                PolicyCheckEvent(
                    turn=state.turn_number,
                    should_consult=should_consult,
                    reason=reason,
                )
            )

            # Check for completion
            done = "[DONE]" in content

            if should_consult:
                logger.info("Consulting advisor: %s", reason)
                context = build_advisor_context(state, reason)
                advisor_response, advisor_model_response = self.advisor.consult(context)

                self.tracker.record("advisor", advisor_model_response)

                state.advisor_calls += 1
                state.advisor_history.append(advisor_response)
                self.policy.record_advisor_call()

                # Inject advisor guidance into conversation
                guidance = (
                    f"[ADVISOR GUIDANCE - turn {state.turn_number}]\n"
                    f"Status: {advisor_response.status}\n"
                    f"Diagnosis: {advisor_response.diagnosis}"
                )
                if advisor_response.recommended_plan:
                    guidance += (
                        f"\nRecommended plan: {advisor_response.recommended_plan}"
                    )
                if advisor_response.next_step:
                    guidance += f"\nNext step: {advisor_response.next_step}"
                if advisor_response.risks:
                    guidance += f"\nRisks: {', '.join(advisor_response.risks)}"

                state.messages.append({"role": "user", "content": guidance})

                self.trace_logger.log(
                    "advisor_call",
                    turn=state.turn_number,
                    reason=reason,
                    advisor_status=advisor_response.status,
                    advisor_diagnosis=advisor_response.diagnosis,
                )

                self._emit(
                    AdvisorCallEvent(
                        turn=state.turn_number,
                        reason=reason,
                        advisor_status=advisor_response.status,
                        diagnosis=advisor_response.diagnosis,
                        recommended_plan=advisor_response.recommended_plan,
                        next_step=advisor_response.next_step,
                        risks=advisor_response.risks,
                        confidence=advisor_response.confidence,
                    )
                )

            self.policy.advance_turn()

            if done:
                final_answer = content.replace("[DONE]", "").strip()
                state.status = "completed"
                logger.info(
                    "Executor signaled completion on turn %d", state.turn_number
                )
                break

        else:
            # Loop exhausted without completion
            state.status = "failed"
            final_answer = next(
                (
                    m["content"]
                    for m in reversed(state.messages)
                    if m["role"] == "assistant" and m["content"].strip()
                ),
                "",
            )
            logger.warning(
                "Executor reached max turns (%d) without completing",
                self.config.max_turns,
            )

        usage = self.tracker.summary()
        self.trace_logger.log(
            "run_complete",
            turns=state.turn_number,
            advisor_calls=state.advisor_calls,
            usage=usage,
        )

        self._emit(
            RunCompleteEvent(
                turns=state.turn_number,
                advisor_calls=state.advisor_calls,
                status=state.status,
                final_answer=final_answer,
                usage=usage,
            )
        )

        return ExecutorResult(
            final_answer=final_answer,
            state=state,
            usage_summary=usage,
            advisor_history=state.advisor_history,
        )
