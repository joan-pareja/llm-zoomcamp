"""Command-line assistant for the monitoring module.

The lesson builds an assistant from two local helpers. This repo keeps that
shared code in ``lib``, so this script wires the same pieces through the common
FAQ loader, search tool, and RAG assistant.
"""

import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.agentic_rag import AgenticRAG  # noqa: E402
from lib.search import MinsearchLexicalSearchTool  # noqa: E402
from lib.search_types import LexicalSearchConfig  # noqa: E402
from lib.sources import load_faq_documents  # noqa: E402
from lib.types import FAQDocument  # noqa: E402

COURSE = "llm-zoomcamp"

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()


def create_assistant() -> AgenticRAG[FAQDocument]:
    """Load the FAQ, build the search index, and return the RAG assistant."""
    config = dotenv_values()
    model = config.get("OPENAI_MODEL_NAME") or "gpt-5.4-mini"
    api_key = config.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env.")

    documents: list[FAQDocument] = load_faq_documents()
    search_tool = MinsearchLexicalSearchTool[FAQDocument].from_documents(
        documents=documents,
        text_fields=["section", "question", "answer"],
        keyword_fields=["course"],
        config=LexicalSearchConfig(
            num_results=5,
            filter_dict={"course": COURSE},
            boost_dict={"question": 3.0, "section": 0.5},
        ),
    )

    return AgenticRAG(
        llm_client=OpenAI(api_key=api_key),
        search_tool=search_tool,
        instructions=INSTRUCTIONS,
        model=model,
        mode="simple",
    )


def main() -> None:
    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    assistant = create_assistant()
    result = assistant.find_and_reply(query)
    print(result.answer)


if __name__ == "__main__":
    main()
