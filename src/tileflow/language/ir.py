"""Structured Python trace IR for TileFlow.

This IR is intentionally MLIR-shaped: operations consume SSA values, produce
SSA values, and may own nested regions. The Python builder records this form
before a later emitter lowers it to real MLIR dialect operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal


MemorySpace = Literal["global", "shared", "local", "fragment", "register", "unknown"]
LoopKind = Literal["for", "sequential", "parallel", "pipelined"]


@dataclass(frozen=True)
class Type:
    """Base IR type."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IndexType(Type):
    def __init__(self) -> None:
        object.__setattr__(self, "name", "index")


@dataclass(frozen=True)
class BoolType(Type):
    def __init__(self) -> None:
        object.__setattr__(self, "name", "bool")


@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool = True

    def __init__(self, bits: int = 32, signed: bool = True) -> None:
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "signed", signed)
        prefix = "i" if signed else "ui"
        object.__setattr__(self, "name", f"{prefix}{bits}")


@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __init__(self, bits: int = 32) -> None:
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "name", f"f{bits}")


@dataclass(frozen=True)
class TensorType(Type):
    shape: tuple[Any, ...]
    element_type: Type | str
    memory_space: MemorySpace = "global"

    def __init__(
        self,
        shape: tuple[Any, ...],
        element_type: Type | str,
        memory_space: MemorySpace = "global",
    ) -> None:
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "memory_space", memory_space)
        shape_text = "x".join(str(dim) for dim in shape) if shape else ""
        object.__setattr__(self, "name", f"tensor<{shape_text}x{element_type}>")


@dataclass
class TensorArg:
    name: str
    index: int
    dtype: str
    rank: int | None = None
    shape: tuple[Any, ...] | None = None
    memory_space: MemorySpace = "global"


@dataclass(frozen=True)
class Value:
    name: str
    type: Type | str
    producer: Operation | None = field(default=None, compare=False, repr=False)

    @property
    def dtype(self) -> str:
        """Backward-compatible spelling used by the bootstrap emitter."""

        return str(self.type)

    def __add__(self, other: ValueLike) -> Value:
        return current_builder().binary("add", self, other)

    def __radd__(self, other: ValueLike) -> Value:
        return current_builder().binary("add", other, self)

    def __sub__(self, other: ValueLike) -> Value:
        return current_builder().binary("sub", self, other)

    def __rsub__(self, other: ValueLike) -> Value:
        return current_builder().binary("sub", other, self)

    def __mul__(self, other: ValueLike) -> Value:
        return current_builder().binary("mul", self, other)

    def __rmul__(self, other: ValueLike) -> Value:
        return current_builder().binary("mul", other, self)

    def __truediv__(self, other: ValueLike) -> Value:
        return current_builder().binary("div", self, other)

    def __floordiv__(self, other: ValueLike) -> Value:
        return current_builder().binary("floordiv", self, other)

    def __mod__(self, other: ValueLike) -> Value:
        return current_builder().binary("mod", self, other)

    def __neg__(self) -> Value:
        return current_builder().unary("neg", self)

    def __and__(self, other: ValueLike) -> Value:
        return current_builder().binary("bitand", self, other)

    def __rand__(self, other: ValueLike) -> Value:
        return current_builder().binary("bitand", other, self)

    def __or__(self, other: ValueLike) -> Value:
        return current_builder().binary("bitor", self, other)

    def __ror__(self, other: ValueLike) -> Value:
        return current_builder().binary("bitor", other, self)

    def __xor__(self, other: ValueLike) -> Value:
        return current_builder().binary("bitxor", self, other)

    def __rxor__(self, other: ValueLike) -> Value:
        return current_builder().binary("bitxor", other, self)

    def __lt__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare("lt", self, other)

    def __le__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare("le", self, other)

    def __gt__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare("gt", self, other)

    def __ge__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare("ge", self, other)


ValueLike = Value | int | float | bool


@dataclass
class Block:
    args: list[Value] = field(default_factory=list)
    ops: list[Operation] = field(default_factory=list)


@dataclass
class Region:
    blocks: list[Block] = field(default_factory=lambda: [Block()])

    @property
    def entry(self) -> Block:
        return self.blocks[0]

    @property
    def ops(self) -> list[Operation]:
        return self.entry.ops


class Operation:
    """An SSA operation with optional nested regions.

    `kind` and single `result` are kept as compatibility aliases for the
    bootstrap code that predates multi-result operations and regions.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        kind: str | None = None,
        results: list[Value] | None = None,
        result: Value | None = None,
        operands: list[Any] | None = None,
        attrs: dict[str, Any] | None = None,
        regions: list[Region] | None = None,
    ) -> None:
        op_name = name if name is not None else kind
        if op_name is None:
            raise TypeError("Operation requires `name` or `kind`")
        self.name = op_name
        self.results = results if results is not None else ([] if result is None else [result])
        self.operands = operands or []
        self.attrs = attrs or {}
        self.regions = regions or []

    @property
    def kind(self) -> str:
        return self.name

    @property
    def result(self) -> Value | None:
        return self.results[0] if self.results else None


@dataclass
class KernelIR:
    name: str
    args: list[TensorArg] = field(default_factory=list)
    body: Region = field(default_factory=Region)

    @property
    def ops(self) -> list[Operation]:
        """Backward-compatible root operation list."""

        return self.body.ops

    def values(self) -> list[Value]:
        values: list[Value] = []

        def visit_region(region: Region) -> None:
            for op in region.ops:
                values.extend(op.results)
                for nested in op.regions:
                    visit_region(nested)

        visit_region(self.body)
        return values


class IRBuilderError(RuntimeError):
    pass


_builder_stack: list[IRBuilder] = []


def current_builder() -> IRBuilder:
    if not _builder_stack:
        raise IRBuilderError("IR operation used outside an active IRBuilder")
    return _builder_stack[-1]


class IRBuilder:
    def __init__(self, name: str):
        self.ir = KernelIR(name=name)
        self._next_value = 0
        self._region_stack: list[Region] = [self.ir.body]

    def __enter__(self) -> IRBuilder:
        _builder_stack.append(self)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        popped = _builder_stack.pop()
        if popped is not self:
            raise IRBuilderError("IRBuilder stack corruption")
        return False

    @property
    def current_region(self) -> Region:
        return self._region_stack[-1]

    @property
    def current_block(self) -> Block:
        return self.current_region.entry

    @contextmanager
    def region(self, region: Region | None = None) -> Iterator[Region]:
        region = region or Region()
        self._region_stack.append(region)
        try:
            yield region
        finally:
            popped = self._region_stack.pop()
            if popped is not region:
                raise IRBuilderError("region stack corruption")

    def argument(
        self,
        name: str,
        index: int,
        dtype: str,
        rank: int | None,
        *,
        shape: tuple[Any, ...] | None = None,
        memory_space: MemorySpace = "global",
    ) -> TensorArg:
        arg = TensorArg(
            name=name,
            index=index,
            dtype=dtype,
            rank=rank,
            shape=shape,
            memory_space=memory_space,
        )
        self.ir.args.append(arg)
        return arg

    def value(self, dtype: Type | str = "index", *, prefix: str = "%v") -> Value:
        value = Value(name=f"{prefix}{self._next_value}", type=dtype)
        self._next_value += 1
        return value

    def emit(
        self,
        kind: str,
        operands: list[Any] | None = None,
        *,
        dtype: Type | str = "index",
        attrs: dict[str, Any] | None = None,
        has_result: bool = True,
        regions: list[Region] | None = None,
        results: list[Value] | None = None,
    ) -> Value | None:
        op_results = results if results is not None else ([self.value(dtype)] if has_result else [])
        op = Operation(
            kind=kind,
            results=op_results,
            operands=operands or [],
            attrs=attrs or {},
            regions=regions or [],
        )
        self.current_block.ops.append(op)
        return op.result

    def emit_op(
        self,
        name: str,
        operands: list[Any] | None = None,
        *,
        result_types: list[Type | str] | None = None,
        attrs: dict[str, Any] | None = None,
        regions: list[Region] | None = None,
    ) -> Operation:
        results = [self.value(dtype) for dtype in (result_types or [])]
        op = Operation(
            name=name,
            results=results,
            operands=operands or [],
            attrs=attrs or {},
            regions=regions or [],
        )
        self.current_block.ops.append(op)
        return op

    def ensure_value(self, value: ValueLike) -> Value:
        if isinstance(value, Value):
            return value
        if isinstance(value, bool):
            return self.const(value, BoolType())
        if isinstance(value, int):
            return self.const(value, IndexType())
        if isinstance(value, float):
            return self.const(value, FloatType(32))
        raise TypeError(f"cannot convert {type(value).__name__} to IR Value")

    def const(self, value: int | float | bool, dtype: Type | str | None = None) -> Value:
        if dtype is None:
            if isinstance(value, bool):
                dtype = BoolType()
            elif isinstance(value, int):
                dtype = IndexType()
            else:
                dtype = FloatType(32)
        result = self.emit("const", attrs={"value": value}, dtype=dtype)
        assert result is not None
        return result

    def unary(self, op: str, operand: ValueLike, *, dtype: Type | str | None = None) -> Value:
        operand_value = self.ensure_value(operand)
        result = self.emit(op, [operand_value], dtype=dtype or operand_value.type)
        assert result is not None
        return result

    def binary(
        self,
        op: str,
        lhs: ValueLike,
        rhs: ValueLike,
        *,
        dtype: Type | str | None = None,
    ) -> Value:
        lhs_value = self.ensure_value(lhs)
        rhs_value = self.ensure_value(rhs)
        result = self.emit(op, [lhs_value, rhs_value], dtype=dtype or lhs_value.type)
        assert result is not None
        return result

    def compare(self, op: str, lhs: ValueLike, rhs: ValueLike) -> Value:
        return self.binary(op, lhs, rhs, dtype=BoolType())

    def cast(self, value: ValueLike, dtype: Type | str) -> Value:
        return self.unary("cast", value, dtype=dtype)

    def select(self, cond: ValueLike, true_value: ValueLike, false_value: ValueLike) -> Value:
        true_ir = self.ensure_value(true_value)
        result = self.emit(
            "select",
            [self.ensure_value(cond), true_ir, self.ensure_value(false_value)],
            dtype=true_ir.type,
        )
        assert result is not None
        return result

    def alloc(
        self,
        name: str,
        shape: tuple[Any, ...],
        dtype: str,
        *,
        memory_space: MemorySpace = "local",
    ) -> Value:
        result = self.emit(
            "alloc",
            attrs={"name": name, "shape": shape, "dtype": dtype, "memory_space": memory_space},
            dtype=TensorType(shape, dtype, memory_space),
        )
        assert result is not None
        return result

    def load(self, tensor: ValueLike, indices: list[ValueLike], *, dtype: Type | str) -> Value:
        operands = [self.ensure_value(tensor)] + [self.ensure_value(index) for index in indices]
        result = self.emit("load", operands, dtype=dtype)
        assert result is not None
        return result

    def store(self, value: ValueLike, tensor: ValueLike, indices: list[ValueLike]) -> None:
        operands = [self.ensure_value(value), self.ensure_value(tensor)] + [
            self.ensure_value(index) for index in indices
        ]
        self.emit("store", operands, has_result=False)

    def if_op(
        self, cond: ValueLike, then_region: Region, else_region: Region | None = None
    ) -> None:
        regions = [then_region] if else_region is None else [then_region, else_region]
        self.emit(
            "if",
            [self.ensure_value(cond)],
            attrs={"has_else": else_region is not None},
            has_result=False,
            regions=regions,
        )

    def while_op(self, cond_region: Region, body_region: Region) -> None:
        self.emit("while", has_result=False, regions=[cond_region, body_region])

    def for_op(
        self,
        lo: ValueLike,
        hi: ValueLike,
        step: ValueLike,
        body_region: Region,
        *,
        kind: LoopKind = "for",
    ) -> Value:
        iv = self.value(IndexType(), prefix="%iv")
        self.emit(
            "for",
            [self.ensure_value(lo), self.ensure_value(hi), self.ensure_value(step)],
            attrs={"kind": kind, "iv": iv},
            has_result=False,
            regions=[body_region],
        )
        return iv

    def yield_op(self, values: list[ValueLike] | None = None) -> None:
        self.emit("yield", [self.ensure_value(value) for value in values or []], has_result=False)

    def break_op(self) -> None:
        self.emit("break", has_result=False)

    def continue_op(self) -> None:
        self.emit("continue", has_result=False)

    def return_op(self, values: list[ValueLike] | None = None) -> None:
        self.emit("return", [self.ensure_value(value) for value in values or []], has_result=False)
