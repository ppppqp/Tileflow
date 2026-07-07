"""Tileflow JIT decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tileflow.compiler.ast import IRGenerator, parse_jit_function
from tileflow.compiler.compiler import JITImplementation
from tileflow.dsl.language import Buffer, Var

"""
jit decorator
converts jit function to a JitImplementation object
"""


def jit(func: Callable):
    def decorator(func: Callable):
        pf: JitFunction = parse_jit_function(func)
        return JITImplementation(func=func, jit_function=pf)

    return decorator(func)


class JitFunction:
    original_func: Callable[..., Any]
    arg_names: list[str]
    tensor_args: dict[str, Buffer | Var]
    ir_generator: IRGenerator

    def __init__(
        self,
        original_func: Callable[..., Any],
        tensor_args: dict[str, Buffer | Var],
        ir_generator: IRGenerator,
        arg_names: list[str],
    ):
        self.original_func = original_func
        self.name = original_func.__name__
        self.tensor_args = tensor_args
        self.ir_generator = ir_generator
        self.arg_names = arg_names

    def __repr__(self):
        return f"<tileflow.jit ({self.name})>"

    def parse_args(self, *args, **kwargs):
        # TODO: phase 2 args (seems tir related?) templates?
        kwargs.update({k: v for k, v in zip(self.arg_names, args, strict=True)})
        tensor_args = {}
        for k in self.tensor_args:
            tensor_args[k] = kwargs.pop(k)
        return tensor_args, kwargs
