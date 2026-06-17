"""Compiler passes for the bootstrap pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from tileflow.ir import KernelIR, Operation
from tileflow.layout import LayoutConstraint, infer_layouts


@dataclass(frozen=True)
class PipelineStage:
    index: int
    op_kinds: tuple[str, ...]
    description: str


def run_layout_inference(ir: KernelIR) -> dict[str, LayoutConstraint]:
    return infer_layouts(ir)


def plan_pipeline(ir: KernelIR) -> list[PipelineStage]:
    """Group memory and compute operations into coarse pipeline stages."""

    memory_ops: list[Operation] = []
    compute_ops: list[Operation] = []
    stores: list[Operation] = []
    for op in ir.ops:
        if op.kind == "load":
            memory_ops.append(op)
        elif op.kind == "store":
            stores.append(op)
        elif op.kind not in {"const", "program_id"}:
            compute_ops.append(op)

    stages: list[PipelineStage] = []
    if memory_ops:
        stages.append(PipelineStage(0, tuple(op.kind for op in memory_ops), "global memory read"))
    if compute_ops:
        stages.append(PipelineStage(len(stages), tuple(op.kind for op in compute_ops), "compute"))
    if stores:
        stages.append(PipelineStage(len(stages), tuple(op.kind for op in stores), "global memory write"))
    return stages

