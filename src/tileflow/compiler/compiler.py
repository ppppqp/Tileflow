"""Compile orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from tileflow.compiler.jit import JitFunction
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
        from tileflow.compiler.builder import Builder
        from tileflow.language.proxy import TensorAnnotation

        call_kwargs = dict(kwargs)
        for name, value in zip(self.jit_function.arg_names, args, strict=False):
            call_kwargs[name] = value
        for name in self.jit_function.arg_names:
            call_kwargs.setdefault(name, None)
        call_kwargs = {
            name: None if isinstance(value, TensorAnnotation) else value
            for name, value in call_kwargs.items()
        }

        with Builder(self.jit_function.name) as builder:
            traced = self.jit_function.ir_generator.generator(builder)
            traced(**call_kwargs)
        return builder.ir

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
        from tileflow.compiler.mlir_emitter import emit_upstream_mlir

        module = emit_upstream_mlir(self.ir)
        if not module.operation.verify():
            raise RuntimeError("emitted TileFlow MLIR failed verification")
        self.mlir = str(module)

    def __call__(self, *args, **kwargs):
        self.torch_function(*args, **kwargs)
