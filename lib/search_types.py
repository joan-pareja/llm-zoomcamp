"""Neutral search contracts shared by index builders and search tools.

Keep this module focused on search-specific interfaces and runtime
configuration. General JSON/document shapes and numeric vector aliases belong
in ``lib.types``.

The index interfaces are structural Protocols on purpose. Concrete indexes come
from third-party libraries like ``minsearch`` and ``sqlitesearch``; they do not
inherit from our classes, but they do expose compatible ``search(...)`` methods.
Using Protocols lets builder functions return a clear search interface without
extra adapter boilerplate or caller-side casts.
"""

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, overload

from .types import EmbeddingVector


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


class LexicalSearchIndex(Protocol):
    """Index interface required by lexical search tools."""

    def search(
        self,
        query: str,
        filter_dict: dict[str, Any] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[IndexDocument]:
        ...


class SemanticSearchIndex(Protocol):
    """Index interface required by semantic search tools."""

    def search(
        self,
        query_vector: EmbeddingVector,
        filter_dict: dict[str, Any] | None = None,
        num_results: int = 10,
    ) -> list[IndexDocument]:
        ...


@dataclass
class LexicalSearchConfig:
    """Runtime settings shared by lexical search tools."""

    num_results: int = 5
    filter_dict: dict[str, Any] | None = None
    boost_dict: dict[str, float] | None = None


@dataclass
class SemanticSearchConfig:
    """Runtime settings shared by semantic search tools."""

    num_results: int = 5
    batch_size: int = 50
    filter_dict: dict[str, Any] | None = None
