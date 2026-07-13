from typing import Any, TypeAlias, cast

from openai import OpenAI

from lib.llm import DEFAULT_MODEL, call_llm

SearchResult: TypeAlias = dict[str, str]

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
        index: Any,
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
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return cast(
            list[SearchResult],
            self.index.search(
                query,
                num_results=num_results,
                boost_dict=boost_dict,
                filter_dict=filter_dict,
            ),
        )

    def build_context(self, search_results: list[SearchResult]) -> str:
        lines: list[str] = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[SearchResult]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(
        self,
        instructions: str,
        user_prompt: str,
        model: str | None = None,
    ):
        model = model or self.model

        message_history: list[Any] = [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt},
        ]

        return call_llm(
            client=self.llm_client,
            messages=message_history,
            model=model,
        )

    def rag(self, question: str) -> str:
        search_results = self.search(question)
        user_prompt = self.build_prompt(question, search_results)
        call = self.llm(INSTRUCTIONS, user_prompt)

        cost_in_dollars = call.metrics.price.total_cost_usd
        answer = call.result.output_text

        return (
            f"The cost to answer this question is: ${cost_in_dollars}.\n\n"
            "The answer from the llm is:\n\n"
            f"{answer}"
        )

    def agentic_rag(self, question: str) -> None:
        pass
