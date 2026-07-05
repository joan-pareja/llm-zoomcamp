"""Shared data-shape and low-level value types.

This module is for repository-wide primitives: the generic document shape,
typed course records, and numeric arrays used outside a single subsystem.

Search-specific contracts live in ``lib.search_types``. ``EmbeddingVector``
stays here because embedders, index storage, and search code all exchange the
same numeric vector representation.
"""

from collections.abc import Mapping
from typing import Any, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt


# Bound for document generics (search tools, AgenticRAG, ...). Concrete
# documents are TypedDicts with concrete per-field types (str, int, ...), not
# uniform-value-type mappings, so the bound only needs string-keyed, item
# access - it deliberately does not constrain field value types the way a
# JSON-value union would (type checkers treat a TypedDict as a
# Mapping[str, object] for this purpose regardless of its declared field
# types, so any narrower value-type bound is unsatisfiable by a TypedDict).
Document: TypeAlias = Mapping[str, object]
EmbeddingVector: TypeAlias = npt.NDArray[np.floating[Any]]


class LessonDocument(TypedDict):
    """Markdown lesson page parsed from the course repository."""

    filename: str
    content: str


class LessonChunk(LessonDocument):
    """Searchable chunk created from a lesson page."""

    start: int


class HomeworkGroundTruthRecord(TypedDict):
    """Evaluation question labeled with the lesson page that should answer it."""

    question: str
    filename: str


class FAQGroundTruthRecord(TypedDict):
    """Evaluation question labeled with the FAQ document that should answer it."""

    question: str
    document: str


class CourseMetadata(TypedDict):
    """Course listing metadata returned by the DataTalks.Club FAQ source."""

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
