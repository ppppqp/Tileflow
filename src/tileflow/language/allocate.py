from __future__ import annotations

from tileflow.language.proxy import TensorValue, _normalize_shape, ShapeType
from tileflow.language.dtypes import dtype
from tileflow.language.ir import MemorySpace


class OutTensor:
    def __init__(self, shape: ShapeType, dtype: dtype):
        self.shape = _normalize_shape(shape)
        self.dtype = dtype


# empty is allocated globally
def empty(shape: ShapeType, dtype: dtype) -> OutTensor:
    return OutTensor(shape, dtype)


def alloc_shared(shape: ShapeType, dtype: dtype) -> TensorValue:
    return _alloc_tensor(shape, dtype, memory_space="shared")


def alloc_fragment(shape: ShapeType, dtype: dtype) -> TensorValue:
    from tileflow.language.ir import current_builder

    normalized_shape = _normalize_shape(shape)
    value = current_builder().empty_tile(normalized_shape, dtype)
    return TensorValue(
        name="",
        value=value,
        shape=normalized_shape,
        element_type=dtype,
    )


def _alloc_tensor(shape: ShapeType, dtype: dtype, *, memory_space: MemorySpace) -> TensorValue:
    from tileflow.language.ir import current_builder

    normalized_shape = _normalize_shape(shape)
    value = current_builder().alloc(
        normalized_shape,
        dtype,
        memory_space=memory_space,
    )
    return TensorValue(
        name="",
        value=value,
        shape=normalized_shape,
        element_type=dtype,
    )
