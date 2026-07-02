"""Shared type aliases used across library modules."""

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONDict: TypeAlias = dict[str, JSONValue]
JSONDocument: TypeAlias = Mapping[str, JSONValue]
EmbeddingVector: TypeAlias = npt.NDArray[np.floating[Any]]
