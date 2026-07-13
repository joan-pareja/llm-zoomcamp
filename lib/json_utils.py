"""Shared JSON conversion helpers."""

from collections.abc import Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


@runtime_checkable
class ModelDumpable(Protocol):
    def model_dump(self, **kwargs: object) -> object: ...


def convert_to_jsonable(value: object) -> JSONValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): convert_to_jsonable(item) for key, item in mapping.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[object], value)
        return [convert_to_jsonable(item) for item in sequence]

    if isinstance(value, ModelDumpable):
        try:
            return convert_to_jsonable(value.model_dump(mode="json"))
        except TypeError:
            return convert_to_jsonable(value.model_dump())

    return str(value)
