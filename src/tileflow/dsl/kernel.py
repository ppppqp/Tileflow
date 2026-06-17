"""Kernel decorator and trace entry points."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tileflow.ir import IRBuilder, KernelIR

from .tensor import Expr, TensorProxy, TensorType

_state = threading.local()


@dataclass(frozen=True)
class Kernel:
    fn: Callable[..., Any]
    name: str
    signature: inspect.Signature


def kernel(fn: Callable[..., Any]) -> Kernel:
    return Kernel(fn=fn, name=fn.__name__, signature=inspect.signature(fn))


def current_builder() -> IRBuilder:
    builder = getattr(_state, "builder", None)
    if builder is None:
        raise RuntimeError("TileFlow DSL operations can only run while tracing @tileflow.kernel")
    return builder


def program_id(axis: int) -> Expr:
    builder = current_builder()
    value = builder.emit("program_id", dtype="index", attrs={"axis": axis})
    assert value is not None
    return Expr(builder, value)


def trace(k: Kernel) -> KernelIR:
    builder = IRBuilder(k.name)
    proxies = []
    for index, param in enumerate(k.signature.parameters.values()):
        annotation = param.annotation
        if not isinstance(annotation, TensorType):
            raise TypeError(
                f"Kernel parameter '{param.name}' must be annotated with tileflow.TensorType"
            )
        arg = builder.argument(param.name, index, annotation.dtype, annotation.rank)
        proxies.append(TensorProxy(builder, arg))

    previous = getattr(_state, "builder", None)
    _state.builder = builder
    try:
        k.fn(*proxies)
    finally:
        _state.builder = previous
    return builder.ir

