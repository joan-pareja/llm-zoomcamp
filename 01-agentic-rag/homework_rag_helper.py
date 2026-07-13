from typing import Protocol, TypeAlias

from openai import OpenAI

from lib.llm import DEFAULT_MODEL, ModelInput, call_llm

SearchResult: TypeAlias = dict[str, str]


class HomeworkSearchIndex(Protocol):
    """Lexical index operations used by this homework helper."""

    def search(
        self,
        query: str,
        *,
        num_results: int,
        boost_dict: dict[str, float],
    ) -> list[SearchResult]: ...


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(
        self,
        index: HomeworkSearchIndex,
        llm_client: OpenAI,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        course: str = "llm-zoomcamp",
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        boost_dict = {"filename": 2.0, "content": 1.0}
        # filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            # filter_dict=filter_dict
        )

    def build_context(self, search_results: list[SearchResult]) -> str:
        lines: list[str] = []

        for doc in search_results:
            lines.append("File: " + doc["filename"])
            lines.append("Content: " + doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[SearchResult]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        input_messages: ModelInput = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        call = call_llm(
            client=self.llm_client,
            messages=input_messages,
            model=self.model,
        )

        print(call.metrics)

        return call.result.output_text

    def rag(self, query: str) -> str:
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
