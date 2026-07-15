"""Structured Python trace IR for TileFlow.

The IR is intentionally close to MLIR: values are SSA values, operations own
results and nested regions, blocks have arguments and terminators, and source
spans can be carried through for diagnostics.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
import threading
from typing import Any, Literal


MemorySpace = Literal["global", "shared", "local", "fragment", "register", "unknown"]
LoopKind = Literal["for", "sequential", "parallel", "pipelined"]


class OpName(StrEnum):
    CONST = "const"
    CAST = "cast"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FLOORDIV = "floordiv"
    MOD = "mod"
    NEG = "neg"
    BITAND = "bitand"
    BITOR = "bitor"
    BITXOR = "bitxor"
    SHL = "shl"
    SHR = "shr"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    AND = "and"
    OR = "or"
    NOT = "not"
    SELECT = "select"
    CALL = "call"
    ALLOC = "alloc"
    LOAD = "load"
    STORE = "store"
    IF = "if"
    WHILE = "while"
    FOR = "for"
    YIELD = "yield"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"


TERMINATORS = {
    OpName.YIELD,
    OpName.BREAK,
    OpName.CONTINUE,
    OpName.RETURN,
}


@dataclass(frozen=True)
class Span:
    filename: str
    line: int
    col: int
    end_line: int
    end_col: int


@dataclass(frozen=True)
class Type:
    def __str__(self) -> str:
        return self.__class__.__name__.removesuffix("Type").lower()


@dataclass(frozen=True)
class IndexType(Type):
    def __str__(self) -> str:
        return "index"


@dataclass(frozen=True)
class BoolType(Type):
    def __str__(self) -> str:
        return "i1"


@dataclass(frozen=True)
class IntType(Type):
    bits: int = 32
    signed: bool = True

    def __str__(self) -> str:
        return f"{'i' if self.signed else 'ui'}{self.bits}"


@dataclass(frozen=True)
class FloatType(Type):
    bits: int = 32

    def __str__(self) -> str:
        return f"f{self.bits}"


@dataclass(frozen=True)
class TensorType(Type):
    shape: tuple[Any, ...]
    element_type: Type
    memory_space: MemorySpace = "global"

    def __str__(self) -> str:
        shape_text = "x".join(str(dim) for dim in self.shape)
        prefix = f"{shape_text}x" if shape_text else ""
        return f"tensor<{prefix}{self.element_type}, {self.memory_space}>"


@dataclass(frozen=True)
class KernelParam:
    source_name: str
    index: int
    value: Value


@dataclass(frozen=True)
class Value:
    id: int
    type: Type
    name_hint: str | None = None
    owner: Operation | Block | None = field(default=None, compare=False, repr=False)
    span: Span | None = None

    @property
    def ir_name(self) -> str:
        if self.name_hint:
            return f"%{self.name_hint}"
        return f"%v{self.id}"

    def __add__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.ADD, self, other)

    def __radd__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.ADD, other, self)

    def __sub__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.SUB, self, other)

    def __rsub__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.SUB, other, self)

    def __mul__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.MUL, self, other)

    def __rmul__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.MUL, other, self)

    def __truediv__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.DIV, self, other)

    def __floordiv__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.FLOORDIV, self, other)

    def __mod__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.MOD, self, other)

    def __neg__(self) -> Value:
        return current_builder().unary(OpName.NEG, self)

    def __and__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.BITAND, self, other)

    def __rand__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.BITAND, other, self)

    def __or__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.BITOR, self, other)

    def __ror__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.BITOR, other, self)

    def __xor__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.BITXOR, self, other)

    def __rxor__(self, other: ValueLike) -> Value:
        return current_builder().binary(OpName.BITXOR, other, self)

    def __lt__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare(OpName.LT, self, other)

    def __le__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare(OpName.LE, self, other)

    def __gt__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare(OpName.GT, self, other)

    def __ge__(self, other: ValueLike) -> Value:  # type: ignore[override]
        return current_builder().compare(OpName.GE, self, other)


ValueLike = Value | int | float | bool
AttrValue = Any


@dataclass
class Block:
    args: list[Value] = field(default_factory=list)
    ops: list[Operation] = field(default_factory=list)
    terminator: Operation | None = None

    def append(self, op: Operation) -> None:
        if self.terminator is not None:
            raise IRBuilderError(
                f"cannot append operation {op.name!s} after terminator {self.terminator.name!s}"
            )
        if op.name in TERMINATORS:
            self.terminator = op
        else:
            self.ops.append(op)


@dataclass
class Region:
    blocks: list[Block] = field(default_factory=lambda: [Block()])

    @property
    def entry(self) -> Block:
        return self.blocks[0]


@dataclass
class Operation:
    name: OpName | str
    operands: list[Value] = field(default_factory=list)
    results: list[Value] = field(default_factory=list)
    attrs: dict[str, AttrValue] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    span: Span | None = None


@dataclass
class KernelIR:
    name: str
    params: list[KernelParam] = field(default_factory=list)
    body: Region = field(default_factory=Region)
    span: Span | None = None

    def values(self) -> list[Value]:
        values: list[Value] = []

        def visit_block(block: Block) -> None:
            values.extend(block.args)
            for op in block.ops:
                values.extend(op.results)
                for nested in op.regions:
                    visit_region(nested)
            if block.terminator is not None:
                values.extend(block.terminator.results)
                for nested in block.terminator.regions:
                    visit_region(nested)

        def visit_region(region: Region) -> None:
            for block in region.blocks:
                visit_block(block)

        visit_region(self.body)
        return values


class IRBuilderError(RuntimeError):
    pass


_builder_local = threading.local()


def _builder_stack() -> list[IRBuilder]:
    stack = getattr(_builder_local, "stack", None)
    if stack is None:
        stack = []
        _builder_local.stack = stack
    return stack


def current_builder() -> IRBuilder:
    stack = _builder_stack()
    if not stack:
        raise IRBuilderError("IR operation used outside an active IRBuilder")
    return stack[-1]


class IRBuilder:
    def __init__(self, name: str, *, span: Span | None = None):
        self.ir = KernelIR(name=name, span=span)
        self._next_value = 0
        self._region_stack: list[Region] = [self.ir.body]

    def __enter__(self) -> IRBuilder:
        _builder_stack().append(self)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        popped = _builder_stack().pop()
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

    def new_value(
        self,
        type_: Type,
        *,
        name_hint: str | None = None,
        owner: Operation | Block | None = None,
        span: Span | None = None,
    ) -> Value:
        value = Value(
            id=self._next_value,
            type=type_,
            name_hint=name_hint,
            owner=owner,
            span=span,
        )
        self._next_value += 1
        return value

    def argument(
        self,
        source_name: str,
        index: int,
        type_: Type,
        *,
        name_hint: str | None = None,
        span: Span | None = None,
    ) -> Value:
        value = self.new_value(
            type_, name_hint=name_hint or source_name, owner=self.current_block, span=span
        )
        self.current_block.args.append(value)
        self.ir.params.append(KernelParam(source_name=source_name, index=index, value=value))
        return value

    def append_op(
        self,
        name: OpName | str,
        operands: list[Value] | None = None,
        *,
        result_types: list[Type] | None = None,
        attrs: dict[str, AttrValue] | None = None,
        regions: list[Region] | None = None,
        span: Span | None = None,
    ) -> Operation:
        op = Operation(
            name=name,
            operands=operands or [],
            attrs=attrs or {},
            regions=regions or [],
            span=span,
        )
        op.results = [self.new_value(type_, owner=op, span=span) for type_ in (result_types or [])]
        self.current_block.append(op)
        return op

    def emit_value(
        self,
        name: OpName | str,
        operands: list[Value] | None = None,
        *,
        type_: Type,
        attrs: dict[str, AttrValue] | None = None,
        regions: list[Region] | None = None,
        span: Span | None = None,
    ) -> Value:
        op = self.append_op(
            name,
            operands,
            result_types=[type_],
            attrs=attrs,
            regions=regions,
            span=span,
        )
        return op.results[0]

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

    def const(self, value: int | float | bool, type_: Type | None = None) -> Value:
        if type_ is None:
            if isinstance(value, bool):
                type_ = BoolType()
            elif isinstance(value, int):
                type_ = IndexType()
            else:
                type_ = FloatType(32)
        return self.emit_value(OpName.CONST, type_=type_, attrs={"value": value})

    def unary(self, op: OpName | str, operand: ValueLike, *, type_: Type | None = None) -> Value:
        operand_value = self.ensure_value(operand)
        return self.emit_value(op, [operand_value], type_=type_ or operand_value.type)

    def binary(
        self,
        op: OpName | str,
        lhs: ValueLike,
        rhs: ValueLike,
        *,
        type_: Type | None = None,
    ) -> Value:
        lhs_value = self.ensure_value(lhs)
        rhs_value = self.ensure_value(rhs)
        return self.emit_value(op, [lhs_value, rhs_value], type_=type_ or lhs_value.type)

    def compare(self, op: OpName | str, lhs: ValueLike, rhs: ValueLike) -> Value:
        return self.binary(op, lhs, rhs, type_=BoolType())

    def cast(self, value: ValueLike, type_: Type) -> Value:
        return self.unary(OpName.CAST, value, type_=type_)

    def select(self, cond: ValueLike, true_value: ValueLike, false_value: ValueLike) -> Value:
        true_ir = self.ensure_value(true_value)
        return self.emit_value(
            OpName.SELECT,
            [self.ensure_value(cond), true_ir, self.ensure_value(false_value)],
            type_=true_ir.type,
        )

    def alloc(
        self,
        shape: tuple[Any, ...],
        element_type: Type,
        *,
        memory_space: MemorySpace = "local",
        name_hint: str | None = None,
    ) -> Value:
        return self.emit_value(
            OpName.ALLOC,
            type_=TensorType(shape, element_type, memory_space),
            attrs={"shape": shape, "memory_space": memory_space, "name_hint": name_hint},
        )

    def load(self, tensor: ValueLike, indices: list[ValueLike], *, type_: Type) -> Value:
        operands = [self.ensure_value(tensor)] + [self.ensure_value(index) for index in indices]
        return self.emit_value(OpName.LOAD, operands, type_=type_)

    def store(self, value: ValueLike, tensor: ValueLike, indices: list[ValueLike]) -> None:
        operands = [self.ensure_value(value), self.ensure_value(tensor)] + [
            self.ensure_value(index) for index in indices
        ]
        self.append_op(OpName.STORE, operands)

    def if_op(
        self, cond: ValueLike, then_region: Region, else_region: Region | None = None
    ) -> None:
        regions = [then_region] if else_region is None else [then_region, else_region]
        self.append_op(
            OpName.IF,
            [self.ensure_value(cond)],
            attrs={"has_else": else_region is not None},
            regions=regions,
        )

    def while_op(self, cond_region: Region, body_region: Region) -> None:
        self.append_op(OpName.WHILE, regions=[cond_region, body_region])

    def yield_op(self, values: list[ValueLike] | None = None) -> None:
        self.append_op(OpName.YIELD, [self.ensure_value(value) for value in values or []])

    def break_op(self) -> None:
        self.append_op(OpName.BREAK)

    def continue_op(self) -> None:
        self.append_op(OpName.CONTINUE)

    def return_op(self, values: list[ValueLike] | None = None) -> None:
        self.append_op(OpName.RETURN, [self.ensure_value(value) for value in values or []])
