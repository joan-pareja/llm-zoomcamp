import os


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")


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
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model=DEFAULT_OPENAI_MODEL
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model


    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
    

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()


    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )


    def _calculate_cost(self, response):
        input_price = 0.75 / 1_000_000
        output_price = 4.50 / 1_000_000

        cost = (
            response.usage.input_tokens * input_price +
            response.usage.output_tokens * output_price
        )
        return round(cost, 6)


    def llm(self, instructions, user_prompt, model=None):
        model = model or self.model

        message_history = [
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_client.responses.create(
            model=model,
            input=message_history
        )

        return response


    def rag(self, question):
        search_results = self.search(question)
        user_prompt = self.build_prompt(question, search_results)
        response = self.llm(INSTRUCTIONS, user_prompt)

        cost_in_dollars = self._calculate_cost(response)
        answer = response.output_text

        return (
            f"The cost to answer this question is: ${cost_in_dollars}.\n\n"
            "The answer from the llm is:\n\n"
            f"{answer}"
        )


    def agentic_rag(self, question):
        pass
