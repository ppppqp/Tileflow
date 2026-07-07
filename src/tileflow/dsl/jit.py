"""Tileflow JIT decorator."""

from __future__ import annotations

from collections.abc import Callable
from tileflow.dsl.ast import parse_jit_function, JitFunction
from tileflow.compiler.compiler import JITImplementation

"""
jit decorator
converts jit function to a JitImplementation object
"""


def jit(func: Callable):
    def decorator(func: Callable):
        pf: JitFunction = parse_jit_function(func)
        return JITImplementation(func=func, jit_function=pf)

    return decorator(func)
