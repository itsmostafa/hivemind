import pytest


@pytest.fixture
def sample_task():
    return "Explain the tradeoffs between REST and GraphQL"


@pytest.fixture
def minimal_config():
    return {}


@pytest.fixture
def policy_config():
    from hivemind.schemas import PolicyConfig

    return PolicyConfig(
        max_advisor_calls=3, failure_threshold=2, stagnation_turns=3, cooldown_turns=1
    )


@pytest.fixture
def home_tmp(tmp_path, monkeypatch):
    """Redirect USER_CONFIG_PATH to a temporary directory for isolation."""
    monkeypatch.setattr(
        "hivemind.config.USER_CONFIG_PATH", tmp_path / ".hivemind" / "config.yml"
    )
    monkeypatch.setattr(
        "hivemind.cli._config.USER_CONFIG_PATH", tmp_path / ".hivemind" / "config.yml"
    )
    return tmp_path
