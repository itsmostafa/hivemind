"""Tests for ModelClient search tool behavior."""

from unittest.mock import MagicMock, patch

from coagent.models import TAVILY_SEARCH_TOOL, ModelClient
from coagent.schemas import ModelConfig, SearchConfig


def _make_client(
    search_enabled: bool = False, api_key: str | None = None
) -> ModelClient:
    config = ModelConfig(model="ollama/llama3")
    search = SearchConfig(enabled=search_enabled, api_key=api_key)
    return ModelClient(config, search=search)


def _mock_response(
    content: str, tool_calls=None, prompt_tokens: int = 10, completion_tokens: int = 5
):
    """Build a fake litellm response object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def test_generate_without_search_no_tools():
    """When search is disabled, litellm is called without tools kwarg."""
    client = _make_client(search_enabled=False)
    plain_response = _mock_response("hello")

    with patch(
        "coagent.models.litellm.completion", return_value=plain_response
    ) as mock_completion:
        with patch("coagent.models.litellm.completion_cost", return_value=0.0):
            client.generate([{"role": "user", "content": "hi"}])

    call_kwargs = mock_completion.call_args[1]
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


def test_generate_with_search_enabled_passes_tool():
    """When search is enabled, litellm is called with tools and tool_choice='auto'."""
    client = _make_client(search_enabled=True)
    plain_response = _mock_response("answer")

    with patch(
        "coagent.models.litellm.completion", return_value=plain_response
    ) as mock_completion:
        with patch("coagent.models.litellm.completion_cost", return_value=0.0):
            client.generate([{"role": "user", "content": "hi"}])

    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["tools"] == [TAVILY_SEARCH_TOOL]
    assert call_kwargs["tool_choice"] == "auto"


def test_generate_model_skips_tool_when_not_needed():
    """Model returns plain content even when search is enabled — no Tavily call occurs."""
    client = _make_client(search_enabled=True)
    plain_response = _mock_response("direct answer")

    with patch("coagent.models.litellm.completion", return_value=plain_response):
        with patch("coagent.models.litellm.completion_cost", return_value=0.0):
            with patch.object(client, "_run_tavily_search") as mock_search:
                result = client.generate(
                    [{"role": "user", "content": "simple question"}]
                )

    mock_search.assert_not_called()
    assert result.content == "direct answer"


def test_generate_handles_tool_call_loop():
    """Model returns tool_call first, then plain content after search result injected."""
    client = _make_client(search_enabled=True)

    # Build a tool call
    tool_call = MagicMock()
    tool_call.id = "call_abc"
    tool_call.type = "function"
    tool_call.function.name = "tavily_search"
    tool_call.function.arguments = '{"query": "latest AI news"}'

    tool_call_response = _mock_response(
        "", tool_calls=[tool_call], prompt_tokens=15, completion_tokens=8
    )
    final_response = _mock_response(
        "Here are the results.", prompt_tokens=20, completion_tokens=12
    )

    with patch(
        "coagent.models.litellm.completion",
        side_effect=[tool_call_response, final_response],
    ):
        with patch("coagent.models.litellm.completion_cost", return_value=0.0):
            with patch.object(
                client, "_run_tavily_search", return_value='{"results": []}'
            ) as mock_search:
                result = client.generate(
                    [{"role": "user", "content": "latest AI news"}]
                )

    mock_search.assert_called_once_with("latest AI news")
    assert result.content == "Here are the results."


def test_generate_accumulates_tokens_across_tool_calls():
    """Token totals sum across both iterations of the tool call loop."""
    client = _make_client(search_enabled=True)

    tool_call = MagicMock()
    tool_call.id = "call_xyz"
    tool_call.type = "function"
    tool_call.function.name = "tavily_search"
    tool_call.function.arguments = '{"query": "test"}'

    first_response = _mock_response(
        "", tool_calls=[tool_call], prompt_tokens=10, completion_tokens=5
    )
    second_response = _mock_response(
        "final answer", prompt_tokens=20, completion_tokens=15
    )

    with patch(
        "coagent.models.litellm.completion",
        side_effect=[first_response, second_response],
    ):
        with patch("coagent.models.litellm.completion_cost", return_value=0.0):
            with patch.object(client, "_run_tavily_search", return_value="{}"):
                result = client.generate([{"role": "user", "content": "test"}])

    assert result.prompt_tokens == 30  # 10 + 20
    assert result.completion_tokens == 20  # 5 + 15


def test_run_tavily_search_uses_config_api_key():
    """TavilyClient is constructed with api_key from SearchConfig when set."""
    client = _make_client(search_enabled=True, api_key="tvly-config-key")

    mock_tavily_client = MagicMock()
    mock_tavily_client.search.return_value = {"results": []}

    with patch(
        "coagent.models.TavilyClient", return_value=mock_tavily_client
    ) as mock_cls:
        client._run_tavily_search("some query")

    mock_cls.assert_called_once_with(api_key="tvly-config-key")
    mock_tavily_client.search.assert_called_once_with("some query")


def test_run_tavily_search_falls_back_to_env():
    """TavilyClient is constructed without api_key when SearchConfig.api_key is None."""
    client = _make_client(search_enabled=True, api_key=None)

    mock_tavily_client = MagicMock()
    mock_tavily_client.search.return_value = {"results": []}

    with patch(
        "coagent.models.TavilyClient", return_value=mock_tavily_client
    ) as mock_cls:
        client._run_tavily_search("another query")

    mock_cls.assert_called_once_with()
    mock_tavily_client.search.assert_called_once_with("another query")
