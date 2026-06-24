from pathlib import Path
from typing import Any


def build_minsearch_text_index(
    documents: list[dict[str, Any]],
    text_fields: list[str],
    keyword_fields: list[str],
) -> Any:
    from minsearch import Index

    index = Index(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
    )
    index.fit(documents)
    return index


def build_sqlite_text_index(
    documents: list[dict[str, Any]],
    text_fields: list[str],
    keyword_fields: list[str],
    db_path: str | Path,
    recreate: bool = True,
) -> Any:
    from sqlitesearch import TextSearchIndex

    index = TextSearchIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

    if recreate:
        index.clear()

    index.fit(documents)
    return index


def build_minsearch_vector_index(
    vectors: Any,
    documents: list[dict[str, Any]],
    keyword_fields: list[str],
) -> Any:
    from minsearch import VectorSearch

    index = VectorSearch(keyword_fields=keyword_fields)
    index.fit(vectors, documents)
    return index


def build_sqlite_vector_index(
    vectors: Any,
    documents: list[dict[str, Any]],
    keyword_fields: list[str],
    db_path: str | Path,
    mode: str = "ivf",
    recreate: bool = True,
) -> Any:
    from sqlitesearch import VectorSearchIndex

    index = VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )

    if recreate:
        index.clear()

    index.fit(vectors, documents)
    return index


def load_sqlite_text_index(
    text_fields: list[str],
    keyword_fields: list[str],
    db_path: str | Path,
) -> Any:
    from sqlitesearch import TextSearchIndex

    return TextSearchIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )


def load_sqlite_vector_index(
    keyword_fields: list[str],
    db_path: str | Path,
    mode: str = "ivf",
) -> Any:
    from sqlitesearch import VectorSearchIndex

    return VectorSearchIndex(
        mode=mode,
        keyword_fields=keyword_fields,
        db_path=str(db_path),
    )
