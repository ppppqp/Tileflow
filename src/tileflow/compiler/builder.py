from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from tileflow.language.ir import IRBuilder, IndexType, OpName, Region, Span, Value, ValueLike
from tileflow.language.loop import ForSpec, make_range_spec
from tileflow.language.proxy import TensorAnnotation, TensorValue
from tileflow.compiler.ast import _empty
from tileflow.language.kernel import Kernel
from tileflow.language.allocate import OutTensor


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
        self._next_param_index = (
            0  # next parameter index for function arguments (used for TensorValue binding)
        )
        self._next_output_index = 0

    empty = _empty

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
        ivs = tuple(self.ir_builder.new_value(IndexType(), owner=body.entry) for _ in spec.dims)
        body.entry.args.extend(ivs)

        with self.ir_builder.region(body):
            # should we differentiate?
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

    @contextmanager
    def ctx_kernel(self, kernel_ctx: Kernel) -> Iterator[Value | tuple[Value, ...]]:
        if not isinstance(kernel_ctx, Kernel):
            raise TypeError(f"expected Kernel context, got {type(kernel_ctx).__name__}")
        if not kernel_ctx.grid:
            raise ValueError("T.Kernel requires at least one grid dimension")

        body = Region()
        ivs = tuple(
            self.ir_builder.new_value(IndexType(), name_hint=f"block_{i}", owner=body.entry)
            for i in range(len(kernel_ctx.grid))
        )
        body.entry.args.extend(ivs)

        with self.ir_builder.region(body):
            yield ivs[0] if len(ivs) == 1 else ivs

        operands = [self.ir_builder.ensure_value(extent) for extent in kernel_ctx.grid]
        attrs: dict[str, Any] = {"rank": len(kernel_ctx.grid)}
        if kernel_ctx.threads is not None:
            attrs["threads"] = kernel_ctx.threads
        if kernel_ctx.cluster_dims is not None:
            attrs["cluster_dims"] = kernel_ctx.cluster_dims
        self.ir_builder.append_op(
            OpName.KERNEL,
            operands,
            attrs=attrs,
            regions=[body],
        )

    def ctx_while(self, cond: Any) -> Iterator[None]:
        if not callable(cond):
            raise TypeError("while condition must be callable")
        cond_region = Region()
        with self.ir_builder.region(cond_region):
            cond_value = self.ir_builder.ensure_value(cond())
            self.ir_builder.yield_op([cond_value])
        body_region = Region()
        with self.ir_builder.region(body_region):
            yield None
        self.ir_builder.while_op(cond_region, body_region)

    def eval(self, value: Any) -> Any:
        return value

    @staticmethod
    def _indices(value: Any) -> list[Any]:
        return list(value) if isinstance(value, tuple) else [value]

    def assign_slice(self, lval: Any, slice: Any, value: Any) -> None:
        if not isinstance(lval, TensorValue):
            raise TypeError(f"subscript assignment requires a tensor, got {type(lval).__name__}")
        self.ir_builder.store(self._as_value(value), lval.value, self._indices(slice))

    def aug_assign(self, op: str, lhs: Any, rhs: Any, *, name: str | None = None) -> Value:
        result = self._binary_from_ast(op, lhs, rhs)
        if name is not None:
            self.bind(name, result)
        return result

    def aug_assign_slice(self, op: str, lval: Any, slice: Any, rval: Any) -> None:
        if not isinstance(lval, TensorValue):
            raise TypeError(f"subscript assignment requires a tensor, got {type(lval).__name__}")
        indices = self._indices(slice)
        current = self.ir_builder.load(lval.value, indices, type_=lval.element_type)
        self.ir_builder.store(self._binary_from_ast(op, current, rval), lval.value, indices)

    def _binary_from_ast(self, op: str, lhs: Any, rhs: Any) -> Value:
        ops = {
            "Add": OpName.ADD,
            "Sub": OpName.SUB,
            "Mult": OpName.MUL,
            "Div": OpName.DIV,
            "FloorDiv": OpName.FLOORDIV,
            "Mod": OpName.MOD,
            "BitAnd": OpName.BITAND,
            "BitOr": OpName.BITOR,
            "BitXor": OpName.BITXOR,
            "LShift": OpName.SHL,
            "RShift": OpName.SHR,
        }
        try:
            return self.ir_builder.binary(ops[op], self._as_value(lhs), self._as_value(rhs))
        except KeyError as exc:
            raise NotImplementedError(f"unsupported augmented assignment operator {op!r}") from exc

    def boolop(self, op: str, left: Any = None, right: Any = None, operand: Any = None) -> Value:
        if op == "Not":
            return self.ir_builder.unary(OpName.NOT, self._as_value(operand))
        rhs = right() if callable(right) else right
        names = {"And": OpName.AND, "Or": OpName.OR}
        if op not in names:
            raise NotImplementedError(f"unsupported boolean operator {op!r}")
        return self.ir_builder.binary(names[op], self._as_value(left), self._as_value(rhs))

    def ifexp(self, cond: Any, then_: Any, else_: Any) -> Value:
        true_value = then_() if callable(then_) else then_
        false_value = else_() if callable(else_) else else_
        return self.ir_builder.select(
            self._as_value(cond), self._as_value(true_value), self._as_value(false_value)
        )

    def ret(self, value: Any = None) -> None:
        values = [] if value is None else list(value) if isinstance(value, tuple) else [value]
        self.ir_builder.return_op([self._as_value(item) for item in values])

    def assert_expr(self, cond: Any, msg: Any = None) -> None:
        attrs = {} if msg is None else {"message": msg}
        self.ir_builder.append_op(OpName.ASSERT, [self.ir_builder.ensure_value(cond)], attrs=attrs)

    @staticmethod
    def _as_value(value: Any) -> Any:
        return value.value if isinstance(value, TensorValue) else value

    def bind(self, name: str, value: Any, annot: Any | None = None) -> Any:

        # handle type annotation, e.g. `A: T.Tensor((N,), T.float32)`
        if isinstance(annot, TensorAnnotation):
            tensor = self._bind_tensor(name, annot)
            if name != "_":
                self.bindings[-1][name] = tensor
            return tensor

        if value is _empty:
            return value
        if value is None or isinstance(value, (tuple, list, int, float, str)):
            if name != "_":
                self.bindings[-1][name] = value
            return value
        res = self.bind_immut(name, value)
        if name != "_":
            self.bindings[-1][name] = res
        return res

    def bind_immut(self, name: str, value: Any) -> Any:
        name_hint = "tmp" if name == "_" else name
        if isinstance(value, OutTensor):
            output = self.ir_builder.output(
                name_hint,
                self._next_output_index,
                TensorAnnotation(value.shape, value.dtype).buffer_type(),
                name_hint=name_hint,
            )
            self._next_output_index += 1
            return TensorValue(
                name=name_hint,
                value=output,
                shape=value.shape,
                element_type=value.dtype,
            )
        return value

    def _bind_tensor(self, name: str, annot: TensorAnnotation) -> TensorValue:
        value = self.ir_builder.argument(
            name,
            self._next_param_index,
            annot.buffer_type(),
            name_hint=name,
        )
        self._next_param_index += 1
        return TensorValue(
            name=name,
            value=value,
            shape=annot.shape,
            element_type=annot.element_type,
        )

    def rval(self, name: str, fallback: Any = None) -> Any:
        for scope in reversed(self.bindings):
            if name in scope:
                return scope[name]
        return fallback

    def unwrap_value(self, value: Any) -> Any:
        return value
