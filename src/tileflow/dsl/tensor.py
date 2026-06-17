"""Tensor types and trace-time tensor proxies."""

from __future__ import annotations

from dataclasses import dataclass

from tileflow.ir import IRBuilder, TensorArg, Value


@dataclass(frozen=True)
class TensorType:
    dtype: str
    rank: int | None = None


class Expr:
    def __init__(self, builder: IRBuilder, value: Value):
        self.builder = builder
        self.value = value

    def __add__(self, other: object) -> "Expr":
        return self._binary("add", other)

    def __radd__(self, other: object) -> "Expr":
        return self._binary("add", other, reverse=True)

    def __sub__(self, other: object) -> "Expr":
        return self._binary("sub", other)

    def __rsub__(self, other: object) -> "Expr":
        return self._binary("sub", other, reverse=True)

    def __mul__(self, other: object) -> "Expr":
        return self._binary("mul", other)

    def __rmul__(self, other: object) -> "Expr":
        return self._binary("mul", other, reverse=True)

    def _binary(self, kind: str, other: object, *, reverse: bool = False) -> "Expr":
        rhs = ensure_expr(self.builder, other)
        operands = [rhs.value, self.value] if reverse else [self.value, rhs.value]
        value = self.builder.emit(kind, operands, dtype=self.value.dtype)
        assert value is not None
        return Expr(self.builder, value)


class TensorProxy:
    def __init__(self, builder: IRBuilder, arg: TensorArg):
        self.builder = builder
        self.arg = arg

    def __getitem__(self, index: object) -> Expr:
        index_expr = ensure_expr(self.builder, index)
        value = self.builder.emit(
            "load",
            [index_expr.value],
            dtype=self.arg.dtype,
            attrs={"tensor": self.arg.name, "rank": self.arg.rank or 1},
        )
        assert value is not None
        return Expr(self.builder, value)

    def __setitem__(self, index: object, value: object) -> None:
        index_expr = ensure_expr(self.builder, index)
        value_expr = ensure_expr(self.builder, value)
        self.builder.emit(
            "store",
            [index_expr.value, value_expr.value],
            attrs={"tensor": self.arg.name, "rank": self.arg.rank or 1},
            has_result=False,
        )


def ensure_expr(builder: IRBuilder, value: object) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        const = builder.emit("const", dtype="i1", attrs={"value": int(value)})
    elif isinstance(value, int):
        const = builder.emit("const", dtype="index", attrs={"value": value})
    elif isinstance(value, float):
        const = builder.emit("const", dtype="f32", attrs={"value": value})
    else:
        raise TypeError(f"Cannot convert {type(value).__name__} to a TileFlow expression")
    assert const is not None
    return Expr(builder, const)

