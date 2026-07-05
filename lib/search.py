"""Search tool implementations shared across course modules."""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from tqdm.auto import tqdm

from .index_storage import (
    build_minsearch_text_index,
    build_minsearch_vector_index,
    build_sqlite_text_index,
    build_sqlite_vector_index,
)
from .search_types import (
    Encoder,
    LexicalSearchConfig,
    LexicalSearchIndex,
    SemanticSearchConfig,
    SemanticSearchIndex,
)
from .types import Document, EmbeddingVector


class SearchTool[TDocument: Document](ABC):
    """Abstract search interface consumed by AgenticRAG."""

    @abstractmethod
    def search(self, query: str) -> list[TDocument]:
        ...


class BaseLexicalSearchTool[TDocument: Document](SearchTool[TDocument]):
    """Base class for text-query indexes with the same search API."""

    def __init__(
        self,
        index: LexicalSearchIndex,
        config: LexicalSearchConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or LexicalSearchConfig()

    def search(self, query: str) -> list[TDocument]:
        """Search by words, optional filters, and field boosts."""
        results = self.index.search(
            query,
            num_results=self.config.num_results,
            boost_dict=self.config.boost_dict,
            filter_dict=self.config.filter_dict,
        )
        return cast(list[TDocument], results)


class MinsearchLexicalSearchTool[TDocument: Document](
    BaseLexicalSearchTool[TDocument]
):
    """Lexical search backed by an in-memory minsearch text index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[TDocument],
        text_fields: list[str],
        keyword_fields: list[str],
        config: LexicalSearchConfig | None = None,
    ) -> "MinsearchLexicalSearchTool[TDocument]":
        """Build an in-memory lexical tool from raw documents."""
        index = build_minsearch_text_index(
            documents=documents,
            text_fields=text_fields,
            keyword_fields=keyword_fields,
        )
        return cls(index=index, config=config)


class SQLiteLexicalSearchTool[TDocument: Document](
    BaseLexicalSearchTool[TDocument]
):
    """Lexical search backed by a persisted SQLite text index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[TDocument],
        text_fields: list[str],
        keyword_fields: list[str],
        db_path: str | Path,
        config: LexicalSearchConfig | None = None,
        recreate: bool = True,
    ) -> "SQLiteLexicalSearchTool[TDocument]":
        """Build a persisted lexical tool from raw documents."""
        index = build_sqlite_text_index(
            documents=documents,
            text_fields=text_fields,
            keyword_fields=keyword_fields,
            db_path=db_path,
            recreate=recreate,
        )
        return cls(index=index, config=config)


class BaseSemanticSearchTool[TDocument: Document](SearchTool[TDocument]):
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
        documents: Sequence[Document],
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

    def search(self, query: str) -> list[TDocument]:
        """Search by vector similarity and optional filters."""
        results = self.index.search(
            self.encode_query(query),
            num_results=self.config.num_results,
            filter_dict=self.config.filter_dict,
        )
        return cast(list[TDocument], results)


class MinsearchSemanticSearchTool[TDocument: Document](
    BaseSemanticSearchTool[TDocument]
):
    """Semantic search backed by an in-memory minsearch vector index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[TDocument],
        encoder: Encoder,
        text_fields: list[str],
        keyword_fields: list[str],
        config: SemanticSearchConfig | None = None,
    ) -> "MinsearchSemanticSearchTool[TDocument]":
        """Embed documents and build an in-memory semantic tool."""
        config = config or SemanticSearchConfig()
        vectors = cls.encode_documents(documents, encoder, text_fields, config)
        index = build_minsearch_vector_index(
            vectors=vectors,
            documents=documents,
            keyword_fields=keyword_fields,
        )
        return cls(index=index, encoder=encoder, config=config)


class SQLiteSemanticSearchTool[TDocument: Document](
    BaseSemanticSearchTool[TDocument]
):
    """Semantic search backed by a persisted SQLite vector index."""

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[TDocument],
        encoder: Encoder,
        text_fields: list[str],
        keyword_fields: list[str],
        db_path: str | Path,
        config: SemanticSearchConfig | None = None,
        vector_mode: str = "ivf",
        recreate: bool = True,
    ) -> "SQLiteSemanticSearchTool[TDocument]":
        """Embed documents and build a persisted semantic tool."""
        config = config or SemanticSearchConfig()
        vectors = cls.encode_documents(documents, encoder, text_fields, config)
        index = build_sqlite_vector_index(
            vectors=vectors,
            documents=documents,
            keyword_fields=keyword_fields,
            db_path=db_path,
            mode=vector_mode,
            recreate=recreate,
        )
        return cls(index=index, encoder=encoder, config=config)


def reciprocal_rank_fusion[TDocument: Document](
    results_lists: Iterable[Iterable[TDocument]],
    key_fields: Sequence[str],
    k: int = 60,
    num_results: int = 5,
) -> list[TDocument]:
    """Fuse ranked result lists by summing reciprocal ranks per key_fields identity.

    key_fields identifies "the same" document across lists so their ranks can
    be combined; full-object equality isn't safe for this since different
    SearchTool backends may return the same document with extra bookkeeping
    fields or different value representations attached.
    """
    scores: dict[Hashable, float] = dict()
    docs: dict[Hashable, TDocument] = dict()

    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_key = tuple(doc[field] for field in key_fields)
            scores[doc_key] = scores.get(doc_key, 0) + 1 / (k + rank)
            docs[doc_key] = doc

    ranked = sorted(scores, key=lambda doc_key: scores[doc_key], reverse=True)
    return [docs[doc_key] for doc_key in ranked[:num_results]]


class HybridSearchTool[TDocument: Document](SearchTool[TDocument]):
    """Combine two search tools with reciprocal rank fusion."""

    def __init__(
        self,
        lexical_search_tool: SearchTool[TDocument],
        semantic_search_tool: SearchTool[TDocument],
        key_fields: Sequence[str],
        k: int = 60,
        num_results: int = 5,
    ) -> None:
        self.lexical_search_tool = lexical_search_tool
        self.semantic_search_tool = semantic_search_tool
        self.key_fields = key_fields
        self.k = k
        self.num_results = num_results

    def search(self, query: str) -> list[TDocument]:
        """Run lexical and semantic retrieval, then fuse both rankings."""
        lexical_search_results = self.lexical_search_tool.search(query)
        semantic_search_results = self.semantic_search_tool.search(query)
        return reciprocal_rank_fusion(
            [lexical_search_results, semantic_search_results],
            key_fields=self.key_fields,
            k=self.k,
            num_results=self.num_results,
        )