"""Compile orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from tileflow.compiler.ast_frontend import parse_jit_function
from tileflow.compiler.mlir import emit_mlir
from tileflow.compiler.passes import PipelineStage, plan_pipeline, run_layout_inference
from tileflow.dsl import JitFunction
from tileflow.ir import KernelIR
from tileflow.layout import LayoutConstraint


@dataclass(frozen=True)
class CompiledKernel:
    name: str
    ir: KernelIR
    layouts: dict[str, LayoutConstraint]
    pipeline: list[PipelineStage]
    mlir: str


def compile(kernel: JitFunction, **params) -> CompiledKernel:
    ir = parse_jit_function(kernel, params)
    layouts = run_layout_inference(ir)
    pipeline = plan_pipeline(ir)
    mlir = emit_mlir(ir, layouts, pipeline)
    return CompiledKernel(
        name=ir.name,
        ir=ir,
        layouts=layouts,
        pipeline=pipeline,
        mlir=mlir,
    )
