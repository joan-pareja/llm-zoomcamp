"""Agentic RAG utilities shared across course modules."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from tqdm.auto import tqdm

from .index_storage import (
    build_minsearch_text_index,
    build_minsearch_vector_index,
    build_sqlite_text_index,
    build_sqlite_vector_index,
)


JSONDict: TypeAlias = dict[str, Any]
ToolDefinition: TypeAlias = dict[str, Any]
Message: TypeAlias = dict[str, Any]
MessageHistory: TypeAlias = list[Any]
EmbeddingInput: TypeAlias = str | list[str]


class Encoder(Protocol):
    """Minimal interface for embedding models used by semantic search."""

    def encode(self, text: EmbeddingInput) -> Any:
        ...


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
class KeywordSearchConfig:
    num_results: int = 5
    boost_dict: dict[str, float] = field(default_factory=lambda: {
        "question": 3.0,
        "section": 0.5,
    })


@dataclass
class SemanticSearchConfig:
    num_results: int = 5
    batch_size: int = 50


@dataclass
class UsageCostConfig:
    input_price_per_million: float = 0.75
    output_price_per_million: float = 4.50


@dataclass
class AgentRunStats:
    cost_in_dollars: float = 0.0
    tool_calls: int = 0


class SearchTool:
    def search(self, query: str) -> list[JSONDict]:
        raise NotImplementedError


class KeywordSearchTool(SearchTool):
    """Keyword-search adapter for course document indexes."""

    def __init__(
        self,
        index: Any,
        course: str = "llm-zoomcamp",
        config: KeywordSearchConfig | None = None,
    ) -> None:
        self.index = index
        self.course = course
        self.config = config or KeywordSearchConfig()

    @classmethod
    def from_documents(
        cls,
        documents: list[JSONDict],
        text_fields: list[str],
        keyword_fields: list[str],
        course: str = "llm-zoomcamp",
        config: KeywordSearchConfig | None = None,
        db_path: str | Path | None = None,
        recreate: bool = True,
    ) -> "KeywordSearchTool":
        if db_path is None:
            index = build_minsearch_text_index(
                documents=documents,
                text_fields=text_fields,
                keyword_fields=keyword_fields,
            )
        else:
            index = build_sqlite_text_index(
                documents=documents,
                text_fields=text_fields,
                keyword_fields=keyword_fields,
                db_path=db_path,
                recreate=recreate,
            )

        return cls(index=index, course=course, config=config)

    def search(self, query: str) -> list[JSONDict]:
        return self.index.search(
            query,
            num_results=self.config.num_results,
            boost_dict=self.config.boost_dict,
            filter_dict={"course": self.course},
        )


class SemanticSearchTool(SearchTool):
    """Semantic-search adapter for course document indexes."""

    def __init__(
        self,
        index: Any,
        encoder: Encoder,
        course: str = "llm-zoomcamp",
        config: SemanticSearchConfig | None = None,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.course = course
        self.config = config or SemanticSearchConfig()

    @classmethod
    def from_documents(
        cls,
        documents: list[JSONDict],
        encoder: Encoder,
        text_fields: list[str],
        keyword_fields: list[str],
        course: str = "llm-zoomcamp",
        config: SemanticSearchConfig | None = None,
        db_path: str | Path | None = None,
        vector_mode: str = "ivf",
        recreate: bool = True,
    ) -> "SemanticSearchTool":
        config = config or SemanticSearchConfig()
        texts = [
            " ".join(str(doc.get(field, "")) for field in text_fields).strip()
            for doc in documents
        ]
        vectors = []

        for i in tqdm(range(0, len(texts), config.batch_size)):
            batch = texts[i:i + config.batch_size]

            if hasattr(encoder, "encode_batch"):
                batch_vectors = encoder.encode_batch(batch)
            else:
                batch_vectors = encoder.encode(batch)

            vectors.extend(batch_vectors)

        vectors = np.asarray(vectors)

        if db_path is None:
            index = build_minsearch_vector_index(
                vectors=vectors,
                documents=documents,
                keyword_fields=keyword_fields,
            )
        else:
            index = build_sqlite_vector_index(
                vectors=vectors,
                documents=documents,
                keyword_fields=keyword_fields,
                db_path=db_path,
                mode=vector_mode,
                recreate=recreate,
            )

        return cls(
            index=index,
            encoder=encoder,
            course=course,
            config=config,
        )

    def encode_query(self, query: str) -> Any:
        return np.asarray(self.encoder.encode(query))

    def search(self, query: str) -> list[JSONDict]:
        return self.index.search(
            self.encode_query(query),
            num_results=self.config.num_results,
            filter_dict={"course": self.course},
        )


class AgenticRAG:
    """Retrieval-augmented answer generator backed by OpenAI function calls."""

    def __init__(
        self,
        llm_client: Any,
        search_tool: SearchTool,
        instructions: str,
        model: str = "gpt-5.4-mini",
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
