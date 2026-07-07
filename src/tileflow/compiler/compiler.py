"""Compile orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from tileflow.language import JitFunction
    from tileflow.language.ir import KernelIR


class JITImplementation:
    def __init__(self, func: Callable, jit_function: JitFunction):
        self.func = func
        self.jit_function = jit_function

    def __call__(self, *args, **kwargs):
        # the actual compilation to native code happens here

        kernel = self.compile(*args, **kwargs)
        # TODO: cache the compiled kernel based on the arguments
        return kernel(*args, **kwargs)

    def get_ir(self, *args, **kwargs) -> KernelIR:
        return self.func(*args, **kwargs)

    def compile(self, *args, **kwargs) -> CompiledKernel:
        return CompiledKernel(
            name=self.jit_function.name,
            ir=self.get_ir(*args, **kwargs),
        )


class CompiledKernel:
    name: str
    ir: KernelIR
    mlir: str
    torch_function: Callable

    def __init__(
        self,
        name: str,
        ir: KernelIR,
    ):
        self.name = name
        self.ir = ir
        self.compile_kernel()

    def compile_kernel(self):
        # _compile_and_create_adapter
        # tilelang.lower
        pass

    def __call__(self, *args, **kwargs):
        self.torch_function(*args, **kwargs)
