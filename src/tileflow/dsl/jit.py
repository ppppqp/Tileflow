"""Tileflow JIT decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class JitFunction:
    fn: Callable[..., Any]

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.name = fn.__name__

    def __repr__(self):
        return f"<tileflow.jit ({self.name})>"


def jit(fn: Callable) -> JitFunction:
    return JitFunction(fn)
