"""Search tool implementations shared across course modules."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast, overload

import numpy as np
from tqdm.auto import tqdm

from .index_storage import (
    build_minsearch_text_index,
    build_minsearch_vector_index,
    build_sqlite_text_index,
    build_sqlite_vector_index,
)
from .types import EmbeddingVector, JSONDict, JSONDocument


EmbeddingInput: TypeAlias = str | list[str]
IndexDocument: TypeAlias = dict[str, Any]


class Encoder(Protocol):
    """Embedding interface used to create semantic index and query vectors."""

    @overload
    def encode(self, text: str) -> EmbeddingVector:
        ...

    @overload
    def encode(self, text: list[str]) -> EmbeddingVector:
        ...

    def encode(self, text: EmbeddingInput) -> EmbeddingVector:
        ...


class LexicalSearchIndex(ABC):
    """Index interface required by lexical search tools."""

    @abstractmethod
    def search(
        self,
        query: str,
        filter_dict: JSONDict | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[IndexDocument]:
        ...


class SemanticSearchIndex(ABC):
    """Index interface required by semantic search tools."""

    @abstractmethod
    def search(
        self,
        query_vector: EmbeddingVector,
        filter_dict: JSONDict | None = None,
        num_results: int = 10,
    ) -> list[IndexDocument]:
        ...


class SearchTool(ABC):
    """Abstract search interface consumed by AgenticRAG."""

    @abstractmethod
    def search(self, query: str) -> list[JSONDict]:
        ...


@dataclass
class LexicalSearchConfig:
    """Runtime settings shared by lexical search tools."""

    num_results: int = 5
    filter_dict: JSONDict | None = None
    boost_dict: dict[str, float] | None = None


@dataclass
class SemanticSearchConfig:
    """Runtime settings shared by semantic search tools."""

    num_results: int = 5
    batch_size: int = 50
    filter_dict: JSONDict | None = None


class BaseLexicalSearchTool(SearchTool):
    """Base class for text-query indexes with the same search API."""

    def __init__(
        self,
        index: LexicalSearchIndex,
        config: LexicalSearchConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or LexicalSearchConfig()

    def search(self, query: str) -> list[JSONDict]:
        """Search by words, optional filters, and field boosts."""
        results = self.index.search(
            query,
            num_results=self.config.num_results,
            boost_dict=self.config.boost_dict,
            filter_dict=self.config.filter_dict,
        )
        return cast(list[JSONDict], results)


class MinsearchLexicalSearchTool(BaseLexicalSearchTool):
    """Lexical search backed by an in-memory minsearch text index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[JSONDocument],
        text_fields: list[str],
        keyword_fields: list[str],
        config: LexicalSearchConfig | None = None,
    ) -> "MinsearchLexicalSearchTool":
        """Build an in-memory lexical tool from raw documents."""
        index = cast(
            LexicalSearchIndex,
            build_minsearch_text_index(
                documents=documents,
                text_fields=text_fields,
                keyword_fields=keyword_fields,
            ),
        )
        return cls(index=index, config=config)


class SQLiteLexicalSearchTool(BaseLexicalSearchTool):
    """Lexical search backed by a persisted SQLite text index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[JSONDocument],
        text_fields: list[str],
        keyword_fields: list[str],
        db_path: str | Path,
        config: LexicalSearchConfig | None = None,
        recreate: bool = True,
    ) -> "SQLiteLexicalSearchTool":
        """Build a persisted lexical tool from raw documents."""
        index = cast(
            LexicalSearchIndex,
            build_sqlite_text_index(
                documents=documents,
                text_fields=text_fields,
                keyword_fields=keyword_fields,
                db_path=db_path,
                recreate=recreate,
            ),
        )
        return cls(index=index, config=config)


class BaseSemanticSearchTool(SearchTool):
    """Base class for vector indexes that search with encoded queries."""

    def __init__(
        self,
        index: SemanticSearchIndex,
        encoder: Encoder,
        config: SemanticSearchConfig | None = None,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.config = config or SemanticSearchConfig()

    @staticmethod
    def encode_documents(
        documents: Sequence[JSONDocument],
        encoder: Encoder,
        text_fields: list[str],
        config: SemanticSearchConfig,
    ) -> EmbeddingVector:
        """Create document vectors from selected text fields in batches."""
        texts = [
            " ".join(str(doc.get(field, "")) for field in text_fields).strip()
            for doc in documents
        ]
        vectors: list[EmbeddingVector] = []

        for i in tqdm(range(0, len(texts), config.batch_size)):
            batch = texts[i:i + config.batch_size]
            vectors.append(np.asarray(encoder.encode(batch)))

        if vectors:
            return np.vstack(vectors)
        else:
            # Keep empty document collections from failing in np.vstack([]).
            return np.asarray([], dtype=float)

    def encode_query(self, query: str) -> EmbeddingVector:
        """Create one query vector with the same encoder as the index."""
        return np.asarray(self.encoder.encode(query))

    def search(self, query: str) -> list[JSONDict]:
        """Search by vector similarity and optional filters."""
        results = self.index.search(
            self.encode_query(query),
            num_results=self.config.num_results,
            filter_dict=self.config.filter_dict,
        )
        return cast(list[JSONDict], results)


class MinsearchSemanticSearchTool(BaseSemanticSearchTool):
    """Semantic search backed by an in-memory minsearch vector index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[JSONDocument],
        encoder: Encoder,
        text_fields: list[str],
        keyword_fields: list[str],
        config: SemanticSearchConfig | None = None,
    ) -> "MinsearchSemanticSearchTool":
        """Embed documents and build an in-memory semantic tool."""
        config = config or SemanticSearchConfig()
        vectors = cls.encode_documents(documents, encoder, text_fields, config)
        index = cast(
            SemanticSearchIndex,
            build_minsearch_vector_index(
                vectors=vectors,
                documents=documents,
                keyword_fields=keyword_fields,
            ),
        )
        return cls(index=index, encoder=encoder, config=config)


class SQLiteSemanticSearchTool(BaseSemanticSearchTool):
    """Semantic search backed by a persisted SQLite vector index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[JSONDocument],
        encoder: Encoder,
        text_fields: list[str],
        keyword_fields: list[str],
        db_path: str | Path,
        config: SemanticSearchConfig | None = None,
        vector_mode: str = "ivf",
        recreate: bool = True,
    ) -> "SQLiteSemanticSearchTool":
        """Embed documents and build a persisted semantic tool."""
        config = config or SemanticSearchConfig()
        vectors = cls.encode_documents(documents, encoder, text_fields, config)
        index = cast(
            SemanticSearchIndex,
            build_sqlite_vector_index(
                vectors=vectors,
                documents=documents,
                keyword_fields=keyword_fields,
                db_path=db_path,
                mode=vector_mode,
                recreate=recreate,
            ),
        )
        return cls(index=index, encoder=encoder, config=config)
