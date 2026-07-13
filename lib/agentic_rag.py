"""Agentic RAG utilities shared across course modules."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, NamedTuple, TypeAlias, cast

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall, ToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput

from .llm import DEFAULT_MODEL, ModelInput, call_llm
from .metrics import AgentRunMetrics, ModelCallMetrics
from .search import SearchTool
from .types import Document

RAGMode: TypeAlias = Literal["agentic", "simple"]

SEARCH_TOOL_DEFINITION: ToolParam = {
    "type": "function",
    "name": "search",
    "description": "Search the document database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the indexed documents.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class AgentRun:
    answer: str
    metrics: AgentRunMetrics
    message_history: ModelInput


class _AnswerResult(NamedTuple):
    answer: str
    message_history: ModelInput
    model_call_metrics: tuple[ModelCallMetrics, ...]
    tool_calls_count: int


class AgenticRAG[TDocument: Document]:
    """Retrieval-augmented answer generator with simple and agentic modes."""

    def __init__(
        self,
        llm_client: OpenAI,
        search_tool: SearchTool[TDocument],
        instructions: str,
        model: str = DEFAULT_MODEL,
        max_turns: int = 10,
        mode: RAGMode = "agentic",
    ) -> None:
        if mode not in ("agentic", "simple"):
            raise ValueError("mode must be either 'agentic' or 'simple'.")

        self.llm_client = llm_client
        self.search_tool = search_tool
        self.instructions = instructions
        self.model = model
        self.max_turns = max_turns
        self.mode = mode

    def _search_and_serialize(
        self,
        call: ResponseFunctionToolCall,
    ) -> FunctionCallOutput:
        if call.name != "search":
            raise ValueError(f"Unsupported tool call: {call.name}")

        arguments: object = json.loads(call.arguments)
        if not isinstance(arguments, Mapping):
            raise ValueError("Search tool arguments must be a JSON object.")

        argument_mapping = cast(Mapping[object, object], arguments)
        query = argument_mapping.get("query")
        if not isinstance(query, str):
            raise ValueError("Search tool argument 'query' must be a string.")

        result = self.search_tool.search(query)

        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, indent=2),
        }

    def _use_tools_until_done(
        self,
        message_history: ModelInput,
    ) -> _AnswerResult:
        model_call_metrics: list[ModelCallMetrics] = []
        tool_calls_count = 0

        for _ in range(self.max_turns):
            model_call = call_llm(
                client=self.llm_client,
                messages=message_history,
                model=self.model,
                tools=[SEARCH_TOOL_DEFINITION],
            )
            model_call_metrics.append(model_call.metrics)
            response = model_call.result

            message_history = [*message_history, *response.output]

            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if function_calls:
                tool_calls_count += len(function_calls)
                function_results = [
                    self._search_and_serialize(call) for call in function_calls
                ]
                message_history = [*message_history, *function_results]
                continue

            if response.output_text:
                return _AnswerResult(
                    answer=response.output_text,
                    message_history=message_history,
                    model_call_metrics=tuple(model_call_metrics),
                    tool_calls_count=tool_calls_count,
                )

            raise RuntimeError(
                "The LLM response did not include a final answer or tool call."
            )

        raise RuntimeError(f"Agent stopped after {self.max_turns} turns.")

    def _answer_with_single_search(
        self,
        question: str,
    ) -> _AnswerResult:
        documents = self.search_tool.search(question)
        serialized_documents = json.dumps(documents, indent=2)
        message_history: ModelInput = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nContext:\n{serialized_documents}"
                ),
            },
        ]

        model_call = call_llm(
            client=self.llm_client,
            messages=message_history,
            model=self.model,
        )
        response = model_call.result
        message_history = [*message_history, *response.output]

        if response.output_text:
            return _AnswerResult(
                answer=response.output_text,
                message_history=message_history,
                model_call_metrics=(model_call.metrics,),
                tool_calls_count=1,
            )

        raise RuntimeError("The LLM response did not include a final answer.")

    def _answer_agentically(
        self,
        question: str,
    ) -> _AnswerResult:
        message_history: ModelInput = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self._use_tools_until_done(message_history)

    def find_and_reply(self, question: str) -> AgentRun:
        started_at = perf_counter()

        if self.mode == "simple":
            answer, message_history, model_call_metrics, tool_calls_count = (
                self._answer_with_single_search(question)
            )
        else:
            answer, message_history, model_call_metrics, tool_calls_count = (
                self._answer_agentically(question)
            )

        return AgentRun(
            answer=answer,
            metrics=AgentRunMetrics(
                model_call_metrics=model_call_metrics,
                tool_calls_count=tool_calls_count,
                duration_seconds=perf_counter() - started_at,
            ),
            message_history=message_history,
        )
