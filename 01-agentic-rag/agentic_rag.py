"""Agentic RAG implementation with iterative tool use."""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias


FAQDocument: TypeAlias = dict[str, Any]
ToolDefinition: TypeAlias = dict[str, Any]
Message: TypeAlias = dict[str, Any]
MessageHistory: TypeAlias = list[Any]


class SearchIndex(Protocol):
    def search(
        self,
        query: str,
        num_results: int,
        boost_dict: dict[str, float],
        filter_dict: dict[str, str],
    ) -> list[FAQDocument]:
        ...


class SearchTool(Protocol):
    def search(self, query: str) -> list[FAQDocument]:
        ...


class FunctionCall(Protocol):
    type: str
    name: str
    arguments: str
    call_id: str


class LLMResponse(Protocol):
    output: list[Any]
    output_text: str


class ResponsesResource(Protocol):
    def create(self, **kwargs: Any) -> LLMResponse:
        ...


class LLMClient(Protocol):
    responses: ResponsesResource


SEARCH_TOOL: ToolDefinition = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}


@dataclass
class SearchConfig:
    num_results: int = 5
    boost_dict: dict[str, float] = field(default_factory=lambda: {
        "question": 3.0,
        "section": 0.5,
    })


class FAQSearchTool:
    """Search adapter for course FAQ indexes."""

    def __init__(
        self,
        index: SearchIndex,
        course: str = "llm-zoomcamp",
        config: SearchConfig | None = None,
    ) -> None:
        self.index = index
        self.course = course
        self.config = config or SearchConfig()

    def search(self, query: str) -> list[FAQDocument]:
        return self.index.search(
            query,
            num_results=self.config.num_results,
            boost_dict=self.config.boost_dict,
            filter_dict={"course": self.course},
        )


class AgenticRAG:
    """Retrieval-augmented answer generator backed by OpenAI function calls."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        search_tool: SearchTool,
        instructions: str,
        model: str = "gpt-5.4-mini",
        max_turns: int = 10,
    ) -> None:
        """
        Initialize the AgenticRAG system.
        
        Args:
            llm_client: OpenAI client for LLM calls.
            search_tool: Object with a search(query) method.
            instructions (str): System instructions for guiding the agent behavior.
            model (str): LLM model to use for responses.
            max_turns (int): Maximum LLM calls before stopping.
        """
        self.llm_client = llm_client
        self.search_tool = search_tool
        self.instructions = instructions
        self.model = model
        self.max_turns = max_turns
        self.last_message_history: MessageHistory = []
    
    def call_llm(
        self,
        messages: MessageHistory,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """
        Call the LLM with messages and optional tools.
        
        Args:
            messages (list): Message history with role and content.
            tools (list, optional): Tool definitions (e.g., SEARCH_TOOL).
        
        Returns:
            Response object from OpenAI API with output field containing
            response choices (either messages or function calls).
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages
        }
        
        if tools:
            kwargs["tools"] = tools
        
        response = self.llm_client.responses.create(**kwargs)
        return response
    
    def _search_and_serialize(self, call: FunctionCall) -> Message:
        """Execute one search call and format its result for the next turn."""
        args = json.loads(call.arguments)
        
        if call.name != "search":
            raise ValueError(f"Unsupported tool call: {call.name}")

        result = self.search_tool.search(**args)
        
        result_json = json.dumps(result, indent=2)
        
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": result_json,
        }
    
    def _use_tools_until_done(self, message_history: MessageHistory) -> tuple[str, MessageHistory]:
        """Run tool calls until the model returns a final answer."""
        for _ in range(self.max_turns):
            response = self.call_llm(
                messages=message_history,
                tools=[SEARCH_TOOL]
            )
            
            message_history = [*message_history, *response.output]

            function_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            if function_calls:
                function_results = [
                    self._search_and_serialize(call)
                    for call in function_calls
                ]
                message_history = [*message_history, *function_results]
                continue

            if response.output_text:
                return response.output_text, message_history

            raise RuntimeError("The LLM response did not include a final answer or tool call.")

        raise RuntimeError(f"Agent stopped after {self.max_turns} turns.")
    
    def find_and_reply(self, question: str) -> str:
        """
        Find relevant information and generate a reply to the user's question.
        
        Initializes the conversation with system instructions and the user's 
        question, then iterates through tool use until the LLM produces a 
        final response.
        
        Args:
            question (str): The user's question to answer.
        
        Returns:
            str: The final answer generated by the LLM.
        
        
        Example:
            >>> from openai import OpenAI
            >>> from sqlitesearch import TextSearchIndex
            >>> 
            >>> search_tool = FAQSearchTool(index)
            >>> client = OpenAI()
            >>> rag = AgenticRAG(client, search_tool, instructions)
            >>> 
            >>> answer = rag.find_and_reply("Can I join the course now?")
            >>> print(answer)
        """
        message_history: MessageHistory = [
            {
                "role": "system",
                "content": self.instructions
            },
            {
                "role": "user",
                "content": question
            },
        ]
        
        answer, message_history = self._use_tools_until_done(message_history)
        self.last_message_history = message_history
        return answer
