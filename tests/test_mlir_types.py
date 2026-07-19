import pytest

from tileflow.compiler.mlir_types import (
    DYNAMIC_DIM,
    _memory_space_number,
    _normalize_shape,
)
from tileflow.language.ir import IndexType, Value


def test_shape_normalization_preserves_static_and_marks_dynamic_dimensions():
    dynamic = Value(0, IndexType())
    assert _normalize_shape((4, dynamic, 16)) == [4, DYNAMIC_DIM, 16]


def test_shape_normalization_rejects_invalid_static_dimensions():
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_shape((4, -2))
    with pytest.raises(TypeError, match="boolean"):
        _normalize_shape((True, 4))


def test_buffer_memory_space_mapping():
    assert _memory_space_number("global") is None
    assert _memory_space_number("shared") == 3
    assert _memory_space_number("local") == 5
    with pytest.raises(ValueError, match="unsupported"):
        _memory_space_number("fragment")
