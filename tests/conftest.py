import pytest


@pytest.fixture
def sample_task():
    return "Explain the tradeoffs between REST and GraphQL"


@pytest.fixture
def minimal_config():
    return {}
