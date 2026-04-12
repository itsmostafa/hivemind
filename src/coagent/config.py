import os
import re
from typing import Any

import yaml

from coagent.schemas import CoagentConfig, ModelConfig


def _expand_env_vars(value: str) -> str:
    """Expand ${ENV_VAR} references in a string value."""

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def _expand_env_vars_in_dict(data: Any) -> Any:
    """Recursively expand ${ENV_VAR} references in all string values of a dict."""
    if isinstance(data, dict):
        return {k: _expand_env_vars_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars_in_dict(item) for item in data]
    elif isinstance(data, str):
        return _expand_env_vars(data)
    else:
        return data


_CONFIG_CANDIDATES = ("config.yaml", "config.yml")


def load_config(path: str | None = None) -> CoagentConfig:
    """Load CoagentConfig from a YAML file.

    If path is None, looks for config.yaml then config.yml in the current
    directory. If neither exists, returns a default config.
    Expands ${ENV_VAR} references in string values.
    Validates with Pydantic.
    """
    if path is None:
        for candidate in _CONFIG_CANDIDATES:
            if os.path.exists(candidate):
                path = candidate
                break

    if path is None:
        return CoagentConfig(
            executor=ModelConfig(model="ollama/llama3"),
            advisor=ModelConfig(model="ollama/llama3"),
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    raw = _expand_env_vars_in_dict(raw)
    return CoagentConfig.model_validate(raw)


def merge_cli_overrides(
    config: CoagentConfig,
    executor: str | None = None,
    advisor: str | None = None,
    executor_api_base: str | None = None,
    advisor_api_base: str | None = None,
) -> CoagentConfig:
    """Return a new CoagentConfig with CLI overrides applied.

    executor and advisor are model strings (e.g. "ollama/llama3").
    executor_api_base and advisor_api_base set the API endpoint for each model.
    None means "no override, keep config value".
    """
    data = config.model_dump()
    if executor is not None:
        data["executor"]["model"] = executor
    if advisor is not None:
        data["advisor"]["model"] = advisor
    if executor_api_base is not None:
        data["executor"]["api_base"] = executor_api_base
    if advisor_api_base is not None:
        data["advisor"]["api_base"] = advisor_api_base
    return CoagentConfig.model_validate(data)
