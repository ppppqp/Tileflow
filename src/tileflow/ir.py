"""Small traced IR used by the bootstrap compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TensorArg:
    name: str
    index: int
    dtype: str
    rank: int | None = None


@dataclass(frozen=True)
class Value:
    name: str
    dtype: str


@dataclass
class Operation:
    kind: str
    result: Value | None
    operands: list[Value] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelIR:
    name: str
    args: list[TensorArg] = field(default_factory=list)
    ops: list[Operation] = field(default_factory=list)

    def values(self) -> list[Value]:
        return [op.result for op in self.ops if op.result is not None]


class IRBuilder:
    def __init__(self, name: str):
        self.ir = KernelIR(name=name)
        self._next_value = 0

    def argument(self, name: str, index: int, dtype: str, rank: int | None) -> TensorArg:
        arg = TensorArg(name=name, index=index, dtype=dtype, rank=rank)
        self.ir.args.append(arg)
        return arg

    def value(self, dtype: str) -> Value:
        value = Value(name=f"%v{self._next_value}", dtype=dtype)
        self._next_value += 1
        return value

    def emit(
        self,
        kind: str,
        operands: list[Value] | None = None,
        *,
        dtype: str = "index",
        attrs: dict[str, Any] | None = None,
        has_result: bool = True,
    ) -> Value | None:
        result = self.value(dtype) if has_result else None
        self.ir.ops.append(
            Operation(kind=kind, result=result, operands=operands or [], attrs=attrs or {})
        )
        return result

