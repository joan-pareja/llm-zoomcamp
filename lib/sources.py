"""Load FAQ source data.

This module keeps the downloaded JSON as plain dictionaries because the rest of
the project uses subscript access, like ``doc["question"]``. ``TypedDict`` gives
the type checker the expected shape of those dictionaries, while ``cast(...)``
marks the untyped ``response.json()`` boundary as trusted. The cast does not
validate data at runtime; use Pydantic or explicit parsing if runtime validation
becomes important.
"""

from typing import TypedDict, cast

import requests


class CourseMetadata(TypedDict):
    course: str
    course_name: str
    path: str
    questions_count: int


class FAQDocument(TypedDict):
    """FAQ document returned by the DataTalks.Club source."""

    id: str
    course: str
    section: str
    question: str
    answer: str


def load_faq_data() -> list[FAQDocument]:
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    response.raise_for_status()
    courses = cast("list[CourseMetadata]", response.json())

    documents: list[FAQDocument] = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses:
        course_url = f"{url_prefix}{course['path']}"
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_documents = cast("list[FAQDocument]", course_response.json())

        documents.extend(course_documents)

    return documents
