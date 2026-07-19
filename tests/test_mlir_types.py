import pytest

from tileflow.compiler.mlir_types import (
    DYNAMIC_DIM,
    MLIRBindingsUnavailable,
    _memory_space_number,
    _normalize_shape,
    to_mlir_type,
)
from tileflow.language.ir import BufferType, FloatType, IndexType, Value


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


def test_missing_mlir_bindings_have_an_actionable_error():
    with pytest.raises(MLIRBindingsUnavailable, match="MLIR_ENABLE_BINDINGS_PYTHON=ON"):
        to_mlir_type(BufferType((16,), FloatType(32)))
