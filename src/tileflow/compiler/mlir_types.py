"""Conversion from TileFlow frontend types to MLIR Python binding types."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from tileflow.language.ir import (
    BoolType,
    BufferType,
    FloatType,
    IndexType,
    IntType,
    TileType,
    Type,
)


DYNAMIC_DIM = -1

# Keep these aligned with mlir::gpu::AddressSpace. Global memory uses the
# default memref memory space and therefore does not need an attribute.
_MEMORY_SPACE_NUMBERS = {
    "shared": 3,
    "local": 5,
}


class MLIRBindingsUnavailable(RuntimeError):
    """Raised when the LLVM build does not provide the MLIR Python bindings."""


def _load_ir() -> Any:
    try:
        from mlir import ir
    except ImportError as exc:
        raise MLIRBindingsUnavailable(
            "MLIR Python bindings are unavailable; rebuild LLVM with "
            "MLIR_ENABLE_BINDINGS_PYTHON=ON and add its Python package to PYTHONPATH"
        ) from exc
    return ir


def _normalize_shape(shape: tuple[Any, ...]) -> list[int]:
    normalized: list[int] = []
    for dim in shape:
        if isinstance(dim, bool):
            raise TypeError("boolean values are not valid shape dimensions")
        if isinstance(dim, int):
            if dim < 0:
                raise ValueError(f"shape dimensions must be non-negative, got {dim}")
            normalized.append(dim)
        else:
            normalized.append(DYNAMIC_DIM)
    return normalized


def _memory_space_number(memory_space: str) -> int | None:
    if memory_space == "global":
        return None
    try:
        return _MEMORY_SPACE_NUMBERS[memory_space]
    except KeyError as exc:
        raise ValueError(f"unsupported buffer memory space {memory_space!r}") from exc


def to_mlir_type(type_: Type, *, context: Any | None = None) -> Any:
    """Convert one frontend type into an ``mlir.ir.Type``.

    A supplied context is made current for the duration of conversion. When it
    is omitted, callers must already have an active MLIR context.
    """

    ir = _load_ir()
    scope = context if context is not None else nullcontext()
    with scope:
        return _convert_type(type_, ir)


def _convert_type(type_: Type, ir: Any) -> Any:
    if isinstance(type_, IndexType):
        return ir.IndexType.get()
    if isinstance(type_, BoolType):
        return ir.IntegerType.get_signless(1)
    if isinstance(type_, IntType):
        constructor = (
            ir.IntegerType.get_signed if type_.signed else ir.IntegerType.get_unsigned
        )
        return constructor(type_.bits)
    if isinstance(type_, FloatType):
        constructors = {
            16: ir.F16Type.get,
            32: ir.F32Type.get,
            64: ir.F64Type.get,
        }
        try:
            return constructors[type_.bits]()
        except KeyError as exc:
            raise ValueError(f"unsupported floating-point width {type_.bits}") from exc
    if isinstance(type_, BufferType):
        element_type = _convert_type(type_.element_type, ir)
        memory_space_number = _memory_space_number(type_.memory_space)
        memory_space = None
        if memory_space_number is not None:
            i64 = ir.IntegerType.get_signless(64)
            memory_space = ir.IntegerAttr.get(i64, memory_space_number)
        return ir.MemRefType.get(
            _normalize_shape(type_.shape),
            element_type,
            memory_space=memory_space,
        )
    if isinstance(type_, TileType):
        return ir.RankedTensorType.get(
            _normalize_shape(type_.shape),
            _convert_type(type_.element_type, ir),
        )
    raise TypeError(f"cannot convert frontend type {type(type_).__name__} to MLIR")
