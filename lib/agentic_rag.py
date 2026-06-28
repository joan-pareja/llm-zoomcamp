"""Agentic RAG utilities shared across course modules."""

import json
import os
from dataclasses import dataclass
from typing import Any, TypeAlias

from .search import SearchTool


ToolDefinition: TypeAlias = dict[str, Any]
Message: TypeAlias = dict[str, Any]
MessageHistory: TypeAlias = list[Any]

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")


SEARCH_TOOL_DEFINITION: ToolDefinition = {
    "type": "function",
    "name": "search",
    "description": "Search the document database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the indexed documents."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

@dataclass
class UsageCostConfig:
    input_price_per_million: float = 0.75
    output_price_per_million: float = 4.50


@dataclass
class AgentRunStats:
    cost_in_dollars: float = 0.0
    tool_calls: int = 0


class AgenticRAG:
    """Retrieval-augmented answer generator backed by OpenAI function calls."""

    def __init__(
        self,
        llm_client: Any,
        search_tool: SearchTool,
        instructions: str,
        model: str = DEFAULT_OPENAI_MODEL,
        max_turns: int = 10,
        usage_cost_config: UsageCostConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.search_tool = search_tool
        self.instructions = instructions
        self.model = model
        self.max_turns = max_turns
        self.usage_cost_config = usage_cost_config or UsageCostConfig()
        self.last_message_history: MessageHistory = []
        self.last_run_stats = AgentRunStats()

    def call_llm(
        self,
        messages: MessageHistory,
        tools: list[ToolDefinition] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
        }

        if tools:
            kwargs["tools"] = tools

        return self.llm_client.responses.create(**kwargs)

    def _calculate_response_cost(self, response: Any) -> float:
        usage = getattr(response, "usage", None)

        if usage is None:
            return 0.0

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

        input_price = self.usage_cost_config.input_price_per_million / 1_000_000
        output_price = self.usage_cost_config.output_price_per_million / 1_000_000

        return input_tokens * input_price + output_tokens * output_price

    def _search_and_serialize(self, call: Any) -> Message:
        args = json.loads(call.arguments)

        if call.name != "search":
            raise ValueError(f"Unsupported tool call: {call.name}")

        result = self.search_tool.search(**args)

        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, indent=2),
        }

    def _use_tools_until_done(
        self,
        message_history: MessageHistory,
    ) -> tuple[str, MessageHistory, AgentRunStats]:
        run_stats = AgentRunStats()

        for _ in range(self.max_turns):
            response = self.call_llm(
                messages=message_history,
                tools=[SEARCH_TOOL_DEFINITION],
            )
            run_stats.cost_in_dollars += self._calculate_response_cost(response)

            message_history = [*message_history, *response.output]

            function_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            if function_calls:
                run_stats.tool_calls += len(function_calls)
                function_results = [
                    self._search_and_serialize(call)
                    for call in function_calls
                ]
                message_history = [*message_history, *function_results]
                continue

            if response.output_text:
                run_stats.cost_in_dollars = round(run_stats.cost_in_dollars, 6)
                return response.output_text, message_history, run_stats

            raise RuntimeError(
                "The LLM response did not include a final answer or tool call."
            )

        raise RuntimeError(f"Agent stopped after {self.max_turns} turns.")

    def find_and_reply(self, question: str) -> str:
        message_history: MessageHistory = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        answer, message_history, run_stats = self._use_tools_until_done(message_history)
        self.last_message_history = message_history
        self.last_run_stats = run_stats

        tool_call_label = "tool call" if run_stats.tool_calls == 1 else "tool calls"

        return (
            f"The cost to answer this question was: ${run_stats.cost_in_dollars}.\n\n"
            f"The agent used {run_stats.tool_calls} {tool_call_label} to answer it.\n\n"
            f"{answer}"
        )
