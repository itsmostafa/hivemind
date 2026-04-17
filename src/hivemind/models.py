import json
import logging
import uuid
from typing import Any

import litellm
from tavily import TavilyClient

from hivemind.schemas import ModelConfig, ModelResponse, SearchConfig

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

TAVILY_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "Search the web for current information on any topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

MAX_TOOL_ITERATIONS = 10


class ModelClient:
    """Thin wrapper around LiteLLM completion() that normalizes responses."""

    def __init__(self, config: ModelConfig, search: SearchConfig | None = None) -> None:
        self.model = config.model
        self.api_base = config.api_base
        self.api_key = config.api_key
        self.search = search

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send messages to the model and return a normalized response.

        If system is provided, prepend it as a {"role": "system", "content": system}
        message at the start of the messages list.
        """
        if system is not None:
            messages = [{"role": "system", "content": system}] + messages

        # Build litellm kwargs
        session_id = str(uuid.uuid4())
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "metadata": {"litellm_session_id": session_id},
        }
        if self.api_base is not None:
            completion_kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            completion_kwargs["api_key"] = self.api_key
        if self.search and self.search.enabled:
            completion_kwargs["tools"] = [TAVILY_SEARCH_TOOL]
            completion_kwargs["tool_choice"] = "auto"
        completion_kwargs.update(kwargs)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        content = ""
        tool_call_events: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = litellm.completion(**completion_kwargs)

            choice = response.choices[0]

            # Extract token usage
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            # Extract cost — litellm stores it in _hidden_params or response_cost
            cost = 0.0
            try:
                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                # completion_cost can fail when the API returns a versioned model snapshot
                # (e.g. "gpt-5.4-mini-2026-03-05") not in litellm's cost map.
                # Fall back to computing cost from the configured model name, which
                # litellm.get_model_info() resolves correctly (handles "openai/" prefix).
                try:
                    model_info = litellm.get_model_info(self.model)
                    cost = prompt_tokens * (
                        model_info.get("input_cost_per_token") or 0.0
                    ) + completion_tokens * (
                        model_info.get("output_cost_per_token") or 0.0
                    )
                except Exception:
                    # Cost calculation not available for this model (e.g. local Ollama)
                    pass
            total_cost += cost

            # If model returned tool calls, execute them and loop
            if choice.message.tool_calls:
                tool_messages: list[dict[str, Any]] = [
                    {
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    }
                ]
                for tc in choice.message.tool_calls:
                    if tc.function.name == "tavily_search":
                        args = json.loads(tc.function.arguments)
                        result = self._run_tavily_search(args["query"])
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            }
                        )
                        tool_call_events.append(
                            {"tool": "tavily_search", "query": args["query"]}
                        )
                completion_kwargs["messages"] = (
                    completion_kwargs["messages"] + tool_messages
                )
                continue

            # Plain content response — done
            content = choice.message.content or ""
            break

        return ModelResponse(
            content=content,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost=total_cost,
            model=self.model,
            tool_calls=tool_call_events,
        )

    def _run_tavily_search(self, query: str) -> str:
        """Execute a Tavily web search and return JSON results."""
        if self.search and self.search.api_key:
            client = TavilyClient(api_key=self.search.api_key)
        else:
            client = TavilyClient()
        result = client.search(query)
        return json.dumps(result)
