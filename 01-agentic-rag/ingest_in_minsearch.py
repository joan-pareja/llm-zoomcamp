from minsearch import Index


def build_index(documents):
    index = Index(
        text_fields=["question", "section", "answer"], keyword_fields=["course"]
    )
    index.fit(documents)
    return index
