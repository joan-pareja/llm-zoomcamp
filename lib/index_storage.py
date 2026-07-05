"""Build and load concrete search indexes.

This module owns backend setup: creating ``minsearch`` / ``sqlitesearch``
indexes, fitting documents, clearing persisted stores, and loading saved
indexes. ``lib.search`` owns the higher-level search tools that callers use.

The helpers return ``LexicalSearchIndex`` / ``SemanticSearchIndex`` Protocols
instead of concrete backend classes. That is intentional: a value such as
``TextSearchIndex(...)`` can satisfy the Protocol by exposing a compatible
``search(...)`` method, without wrapper adapters or caller-side casts.
"""

from collections.abc import Sequence
from pathlib import Path

from .search_types import LexicalSearchIndex, SemanticSearchIndex
from .types import Document, EmbeddingVector


def build_minsearch_text_index(
    documents: Sequence[Document],
    text_fields: list[str],
    keyword_fields: list[str],
) -> LexicalSearchIndex:
    from minsearch import Index  # pyright: ignore[reportMissingTypeStubs]

    index = Index(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
    )
    index.fit(documents)  # pyright: ignore[reportUnknownMemberType]
    return index


def build_sqlite_text_index(
    documents: Sequence[Document],
    text_fields: list[str],
    keyword_fields: list[str],
    db_path: str | Path,
    recreate: bool = True,
) -> LexicalSearchIndex:
    from sqlitesearch import TextSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    index = TextSearchIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

    if recreate:
        index.clear()  # pyright: ignore[reportUnknownMemberType]

    index.fit(documents)  # pyright: ignore[reportArgumentType]
    return index


def build_minsearch_vector_index(
    vectors: EmbeddingVector,
    documents: Sequence[Document],
    keyword_fields: list[str],
) -> SemanticSearchIndex:
    from minsearch import VectorSearch  # pyright: ignore[reportMissingTypeStubs]

    index = VectorSearch(keyword_fields=keyword_fields)
    index.fit(vectors, documents)  # pyright: ignore[reportUnknownMemberType]
    return index


def build_sqlite_vector_index(
    vectors: EmbeddingVector,
    documents: Sequence[Document],
    keyword_fields: list[str],
    db_path: str | Path,
    mode: str = "ivf",
    recreate: bool = True,
) -> SemanticSearchIndex:
    from sqlitesearch import VectorSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

    if recreate:
        index.clear()  # pyright: ignore[reportUnknownMemberType]

    index.fit(vectors, documents)  # pyright: ignore[reportArgumentType]
    return index


def load_sqlite_text_index(
    text_fields: list[str],
    keyword_fields: list[str],
    db_path: str | Path,
) -> LexicalSearchIndex:
    from sqlitesearch import TextSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    return TextSearchIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )


def load_sqlite_vector_index(
    keyword_fields: list[str],
    db_path: str | Path,
    mode: str = "ivf",
) -> SemanticSearchIndex:
    from sqlitesearch import VectorSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    return VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

