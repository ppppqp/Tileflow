"""Compile orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from tileflow.compiler.ast_frontend import parse_jit_function
from tileflow.compiler.mlir import emit_mlir
from tileflow.compiler.native import NativePipelineResult, run_native_pipeline
from tileflow.dsl import JitFunction
from tileflow.ir import KernelIR


@dataclass(frozen=True)
class CompiledKernel:
    name: str
    ir: KernelIR
    mlir: str
    raw_mlir: str
    native: NativePipelineResult


def compile(
    kernel: JitFunction,
    *,
    mlir_pipeline: str | None = None,
    **params,
) -> CompiledKernel:
    ir = parse_jit_function(kernel, params)
    raw_mlir = emit_mlir(ir)
    native = run_native_pipeline(raw_mlir, pipeline=mlir_pipeline)
    return CompiledKernel(
        name=ir.name,
        ir=ir,
        mlir=native.mlir,
        raw_mlir=raw_mlir,
        native=native,
    )
