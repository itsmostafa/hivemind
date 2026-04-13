"""Tests for DecisionPolicy in coagent.policy."""

from coagent.policy import DecisionPolicy
from coagent.schemas import ExecutorState


def make_state(advisor_calls: int = 0) -> ExecutorState:
    return ExecutorState(task="test task", advisor_calls=advisor_calls)


def test_no_trigger_returns_false(policy_config):
    policy = DecisionPolicy(policy_config)
    state = make_state()
    should, reason = policy.should_consult(state, "Here is a normal response.")
    assert should is False
    assert reason == "no_trigger_met"


def test_explicit_request_triggers(policy_config):
    policy = DecisionPolicy(policy_config)
    state = make_state()
    should, reason = policy.should_consult(state, "I need help. [NEED_ADVICE]")
    assert should is True
    assert reason == "explicit_request"


def test_budget_gate(policy_config):
    # advisor_calls already at max
    policy = DecisionPolicy(policy_config)
    state = make_state(advisor_calls=policy_config.max_advisor_calls)
    should, reason = policy.should_consult(state, "[NEED_ADVICE]")
    assert should is False
    assert reason == "advisor_budget_exhausted"


def test_cooldown_gate(policy_config):
    # policy_config has cooldown_turns=1
    # After record_advisor_call, _turns_since_advisor=0, which is < 1
    policy = DecisionPolicy(policy_config)
    state = make_state()

    # First call goes through fine (starts at 999)
    policy.record_advisor_call()  # reset to 0 — now in cooldown

    # Before advancing the turn, cooldown applies
    should, reason = policy.should_consult(state, "[NEED_ADVICE]")
    assert should is False
    assert reason == "cooldown"


def test_failure_detection(policy_config):
    # failure_threshold=2, so two "i cannot" responses should trigger
    policy = DecisionPolicy(policy_config)
    state = make_state()

    # First failure — below threshold
    should1, _ = policy.should_consult(state, "I cannot do this.")
    assert should1 is False

    # Second failure — at threshold
    should2, reason2 = policy.should_consult(state, "I cannot proceed with that.")
    assert should2 is True
    assert reason2 == "consecutive_failures"


def test_stagnation_detection(policy_config):
    # stagnation_turns=3, so 3 near-identical responses should trigger
    policy = DecisionPolicy(policy_config)
    state = make_state()

    repeated = "The answer is forty two and nothing else here."
    # First two calls accumulate without triggering
    policy.should_consult(state, repeated)
    policy.should_consult(state, repeated)
    # Third identical response should hit stagnation
    should, reason = policy.should_consult(state, repeated)
    assert should is True
    assert reason == "stagnation"


def test_record_advisor_call_resets_cooldown(policy_config):
    # cooldown_turns=1: after record_advisor_call, need 1 advance_turn to exit cooldown
    policy = DecisionPolicy(policy_config)
    state = make_state()

    policy.record_advisor_call()  # _turns_since_advisor = 0

    # Still in cooldown
    should, reason = policy.should_consult(state, "[NEED_ADVICE]")
    assert should is False
    assert reason == "cooldown"

    # Advance one turn — _turns_since_advisor = 1, which equals cooldown_turns=1, no longer < 1
    policy.advance_turn()

    should, reason = policy.should_consult(state, "[NEED_ADVICE]")
    assert should is True
    assert reason == "explicit_request"


def test_advance_turn_increments_counter(policy_config):
    policy = DecisionPolicy(policy_config)
    # Initial value is 999 (high so no cooldown at start)
    policy.record_advisor_call()  # reset to 0
    assert policy._turns_since_advisor == 0

    policy.advance_turn()
    assert policy._turns_since_advisor == 1

    policy.advance_turn()
    assert policy._turns_since_advisor == 2


def test_force_consult_triggers_on_first_call():
    from coagent.schemas import PolicyConfig
    config = PolicyConfig(
        max_advisor_calls=3,
        failure_threshold=2,
        stagnation_turns=3,
        cooldown_turns=1,
        force_consult=True,
    )
    policy = DecisionPolicy(config)
    state = make_state()
    should, reason = policy.should_consult(state, "A normal response.")
    assert should is True
    assert reason == "force_consult"


def test_force_consult_fires_once_then_normal_policy():
    from coagent.schemas import PolicyConfig
    config = PolicyConfig(
        max_advisor_calls=3,
        failure_threshold=2,
        stagnation_turns=3,
        cooldown_turns=1,
        force_consult=True,
    )
    policy = DecisionPolicy(config)
    state = make_state()

    # First call: forced
    should1, reason1 = policy.should_consult(state, "A normal response.")
    assert should1 is True
    assert reason1 == "force_consult"

    # Simulate the advisor call + turn advance so cooldown doesn't mask next check
    policy.record_advisor_call()
    policy.advance_turn()

    # Second call: no trigger — normal response, no heuristic fires
    should2, reason2 = policy.should_consult(state, "A normal response.")
    assert should2 is False
    assert reason2 == "no_trigger_met"


def test_force_consult_bypasses_cooldown_gate():
    from coagent.schemas import PolicyConfig
    config = PolicyConfig(
        max_advisor_calls=3,
        failure_threshold=2,
        stagnation_turns=3,
        cooldown_turns=1,
        force_consult=True,
    )
    policy = DecisionPolicy(config)
    state = make_state()

    # Put policy into cooldown
    policy.record_advisor_call()  # _turns_since_advisor = 0, cooldown active

    # force_consult should still fire despite cooldown
    should, reason = policy.should_consult(state, "A normal response.")
    assert should is True
    assert reason == "force_consult"


def test_force_consult_bypasses_budget_gate():
    from coagent.schemas import PolicyConfig
    config = PolicyConfig(
        max_advisor_calls=3,
        failure_threshold=2,
        stagnation_turns=3,
        cooldown_turns=1,
        force_consult=True,
    )
    policy = DecisionPolicy(config)
    # Exhaust the advisor budget
    state = make_state(advisor_calls=3)

    # force_consult should still fire despite exhausted budget
    should, reason = policy.should_consult(state, "A normal response.")
    assert should is True
    assert reason == "force_consult"


def test_force_consult_false_does_not_affect_normal_policy(policy_config):
    # Sanity: default force_consult=False means no change to existing behavior
    policy = DecisionPolicy(policy_config)
    state = make_state()
    should, reason = policy.should_consult(state, "A normal response.")
    assert should is False
    assert reason == "no_trigger_met"
