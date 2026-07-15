from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from tileflow.language.ir import IRBuilder, IndexType, OpName, Region, Span, Value, ValueLike
from tileflow.language.loop import ForSpec, make_range_spec
from tileflow.compiler.ast import _empty


@dataclass
class IfFrame:
    cond: Value
    then_region: Region
    else_region: Region
    has_else: bool = False


class Builder:
    """Runtime support for the AST-rewritten TileFlow DSL.

    `compiler.ast.DSLMutator` rewrites Python syntax into calls on this object.
    This class translates those calls into structured operations on
    `language.ir.IRBuilder`.
    """

    def __init__(self, name: str, span: Span | None = None):
        self.ir_builder = IRBuilder(name, span=span)
        self.bindings: list[dict[str, Any]] = [
            {}
        ]  # stack of variable bindings for the current scope

    @property
    def ir(self):
        return self.ir_builder.ir

    def __enter__(self) -> Builder:
        self.ir_builder.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return self.ir_builder.__exit__(exc_type, exc, traceback)

    def override(self, name: str):
        # see ast.py
        # we override python builtins like `range` to return TileFlow loop specs instead of Python iterators
        if name == "range":
            return self.range
        raise KeyError(f"no TileFlow override for {name!r}")

    def range(self, *args: ValueLike) -> ForSpec:
        return make_range_spec(*args)

    def normalize_for_spec(self, value: Any) -> ForSpec:
        if isinstance(value, ForSpec):
            return value
        raise TypeError(f"unsupported loop iterator: {type(value).__name__}")

    def ctx_for(self, spec: Any) -> Iterator[Value | tuple[Value, ...]]:
        spec = self.normalize_for_spec(spec)
        body = Region()
        ivs = tuple(self.ir_builder.new_value(IndexType(), name_hint="iv") for _ in spec.dims)
        body.entry.args.extend(ivs)

        with self.ir_builder.region(body):
            yield ivs[0] if len(ivs) == 1 else ivs

        operands: list[Value] = []
        for dim in spec.dims:
            operands.extend(
                [
                    self.ir_builder.ensure_value(dim.lo),
                    self.ir_builder.ensure_value(dim.hi),
                    self.ir_builder.ensure_value(dim.step),
                ]
            )
        self.ir_builder.append_op(
            OpName.FOR,
            operands,
            attrs={
                "kind": spec.kind,
                "ivs": ivs,
                "rank": len(spec.dims),
                **spec.attrs,
            },
            regions=[body],
        )

    def ctx_if(self, cond: ValueLike) -> Iterator[IfFrame]:
        frame = IfFrame(
            cond=self.ir_builder.ensure_value(cond),
            then_region=Region(),
            else_region=Region(),
        )
        yield frame
        self.ir_builder.if_op(
            frame.cond,
            frame.then_region,
            frame.else_region if frame.has_else else None,
        )

    def ctx_then(self, frame: IfFrame) -> Iterator[None]:
        with self.ir_builder.region(frame.then_region):
            yield None

    def ctx_else(self, frame: IfFrame) -> Iterator[None]:
        frame.has_else = True
        with self.ir_builder.region(frame.else_region):
            yield None

    def bind(self, name: str, value: Any) -> Any:

        # handle type annotation, e.g. `A: T.Tensor((N,), T.float32)`
        if value is _empty:
            return self.ir_builder.new_value
        if name != "_":
            self.bindings[-1][name] = value
        return value

    def rval(self, name: str, fallback: Any = None) -> Any:
        for scope in reversed(self.bindings):
            if name in scope:
                return scope[name]
        return fallback

    def unwrap_value(self, value: Any) -> Any:
        return value
