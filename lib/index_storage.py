from collections.abc import Sequence
from pathlib import Path

from .types import EmbeddingVector, JSONDocument


def build_minsearch_text_index(
    documents: Sequence[JSONDocument],
    text_fields: list[str],
    keyword_fields: list[str],
) -> object:
    from minsearch import Index  # pyright: ignore[reportMissingTypeStubs]

    index = Index(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
    )
    index.fit(documents)  # pyright: ignore[reportUnknownMemberType]
    return index


def build_sqlite_text_index(
    documents: Sequence[JSONDocument],
    text_fields: list[str],
    keyword_fields: list[str],
    db_path: str | Path,
    recreate: bool = True,
) -> object:
    from sqlitesearch import TextSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    index = TextSearchIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

    if recreate:
        index.clear()

    index.fit(documents)  # pyright: ignore[reportArgumentType]
    return index


def build_minsearch_vector_index(
    vectors: EmbeddingVector,
    documents: Sequence[JSONDocument],
    keyword_fields: list[str],
) -> object:
    from minsearch import VectorSearch  # pyright: ignore[reportMissingTypeStubs]

    index = VectorSearch(keyword_fields=keyword_fields)
    index.fit(vectors, documents)  # pyright: ignore[reportUnknownMemberType]
    return index


def build_sqlite_vector_index(
    vectors: EmbeddingVector,
    documents: Sequence[JSONDocument],
    keyword_fields: list[str],
    db_path: str | Path,
    mode: str = "ivf",
    recreate: bool = True,
) -> object:
    from sqlitesearch import VectorSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

    if recreate:
        index.clear()

    index.fit(vectors, documents)  # pyright: ignore[reportArgumentType]
    return index


def load_sqlite_text_index(
    text_fields: list[str],
    keyword_fields: list[str],
    db_path: str | Path,
) -> object:
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
) -> object:
    from sqlitesearch import VectorSearchIndex  # pyright: ignore[reportMissingTypeStubs]

    return VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )
