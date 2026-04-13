import re

from coagent.schemas import ExecutorState, PolicyConfig


def _response_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two strings (Jaccard). Returns 0.0-1.0."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class DecisionPolicy:
    """Decides when the executor should consult the advisor."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self._consecutive_failures: int = 0
        self._turns_since_advisor: int = (
            999  # start high so first turn isn't in cooldown
        )
        self._recent_responses: list[str] = []  # last N responses for stagnation check
        self._force_consult_pending: bool = config.force_consult

    def should_consult(
        self, state: ExecutorState, executor_response: str
    ) -> tuple[bool, str]:
        """Evaluate whether the advisor should be consulted after this executor turn.

        Returns (should_consult: bool, reason: str).
        reason explains the trigger (or why no trigger was met).
        """
        # --- Force-consult (one-shot, checked before gates) ---
        if self._force_consult_pending:
            self._force_consult_pending = False
            return True, "force_consult"

        # --- Gates (checked first) ---
        if state.advisor_calls >= self.config.max_advisor_calls:
            return False, "advisor_budget_exhausted"

        if self._turns_since_advisor < self.config.cooldown_turns:
            return False, "cooldown"

        # --- Heuristics (first match wins) ---

        # 1. Explicit request
        if "[NEED_ADVICE]" in executor_response:
            return True, "explicit_request"

        # 2. Failure detection
        lower = executor_response.lower()
        failure_signals = [
            "i cannot",
            "i can't",
            "unable to",
            "i don't know",
            "error:",
            "failed to",
        ]
        if any(sig in lower for sig in failure_signals):
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        if self._consecutive_failures >= self.config.failure_threshold:
            return True, "consecutive_failures"

        # 3. Confidence threshold — executor can self-report with [CONFIDENCE:0.3]
        confidence_match = re.search(r"\[CONFIDENCE:([\d.]+)\]", executor_response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if confidence < self.config.confidence_threshold:
                    return True, "low_confidence"
            except ValueError:
                pass

        # 4. Stagnation detection
        self._recent_responses.append(executor_response)
        if len(self._recent_responses) > self.config.stagnation_turns:
            self._recent_responses.pop(0)

        if len(self._recent_responses) >= self.config.stagnation_turns:
            # Check if all recent responses are very similar to each other
            similarities = []
            for i in range(len(self._recent_responses) - 1):
                sim = _response_similarity(
                    self._recent_responses[i], self._recent_responses[i + 1]
                )
                similarities.append(sim)
            avg_similarity = sum(similarities) / len(similarities)
            if avg_similarity > 0.7:  # 70% word overlap = stagnation
                return True, "stagnation"

        return False, "no_trigger_met"

    def record_advisor_call(self) -> None:
        """Call after each advisor invocation to reset cooldown and failure counters."""
        self._turns_since_advisor = 0
        self._consecutive_failures = 0

    def advance_turn(self) -> None:
        """Call at end of each executor turn to increment cooldown counter."""
        self._turns_since_advisor += 1
