from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tileflow.language.ir import BufferType, TileType, Type, Value
from tileflow.typing import ShapeType, dtype


def _normalize_shape(shape: ShapeType) -> tuple[Any, ...]:
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        return tuple(shape)
    return (shape,)


@dataclass(frozen=True)
class TensorAnnotation:
    shape: tuple[Any, ...]
    element_type: Type

    def buffer_type(self) -> BufferType:
        return BufferType(self.shape, self.element_type)


@dataclass(frozen=True)
class TensorValue:
    name: str
    value: Value
    shape: tuple[Any, ...]
    element_type: Type

    @property
    def type(self) -> BufferType | TileType:
        if not isinstance(self.value.type, (BufferType, TileType)):
            raise TypeError(f"tensor value has non-tensor IR type {self.value.type}")
        return self.value.type

    def __getitem__(self, indices: Any) -> Value:
        from tileflow.language.ir import current_builder

        normalized = list(indices) if isinstance(indices, tuple) else [indices]
        return current_builder().load(self.value, normalized, type_=self.element_type)


class TensorProxy:
    def __call__(self, shape: ShapeType, dtype: dtype) -> TensorAnnotation:
        return TensorAnnotation(shape=_normalize_shape(shape), element_type=dtype)


Tensor = TensorProxy()
