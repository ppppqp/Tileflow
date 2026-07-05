"""Tileflow JIT decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tileflow.compiler.ast import IRGenerator


class JitFunction:
    original_func: Callable[..., Any]

    def __init__(
        self,
        original_func: Callable[..., Any],
        tensor_args: dict[str, Buffer | Var],
        ir_generator: IRGenerator,
    ):
        self.original_func = original_func
        self.name = original_func.__name__

    def __repr__(self):
        return f"<tileflow.jit ({self.name})>"


def jit(fn: Callable) -> JitFunction:
    return JitFunction(fn)
