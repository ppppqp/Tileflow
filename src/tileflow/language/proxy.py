from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tileflow.language.ir import TensorType, Type, Value
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

    def tensor_type(self) -> TensorType:
        return TensorType(self.shape, self.element_type)


@dataclass(frozen=True)
class TensorValue:
    name: str
    value: Value
    shape: tuple[Any, ...]
    element_type: Type

    @property
    def type(self) -> TensorType:
        return TensorType(self.shape, self.element_type)


class TensorProxy:
    def __call__(self, shape: ShapeType, dtype: dtype) -> TensorAnnotation:
        return TensorAnnotation(shape=_normalize_shape(shape), element_type=dtype)


Tensor = TensorProxy()
