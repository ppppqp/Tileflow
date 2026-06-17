"""TileLang-compatible JIT decorator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JitFunction:
    fn: Callable[..., Any]
    target: str | None = None
    compile_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.fn.__name__

    def compile(self, **params: Any):
        from tileflow.compiler import compile

        self.compile_history.append(dict(params))
        return compile(self, **params)


def jit(fn: Callable[..., Any] | None = None, *, target: str | None = None):
    if fn is None:
        return lambda real_fn: JitFunction(real_fn, target=target)
    return JitFunction(fn, target=target)

