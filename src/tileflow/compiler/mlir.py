"""Textual MLIR emitter for the bootstrap IR."""

from __future__ import annotations

from tileflow.compiler.passes import PipelineStage
from tileflow.ir import KernelIR, Operation, Value
from tileflow.layout import LayoutConstraint


def emit_mlir(
    ir: KernelIR,
    layouts: dict[str, LayoutConstraint],
    stages: list[PipelineStage],
) -> str:
    args = ", ".join(f"%{arg.name}: memref<*x{arg.dtype}>" for arg in ir.args)
    lines = [
        "module {",
        f"  func.func @{ir.name}({args}) {{",
    ]
    for name, constraint in layouts.items():
        layout = constraint.layout
        lines.append(
            "    // layout "
            f"{name}: shape={layout.shape}, stride={layout.stride}, space={layout.memory_space}"
        )
    for stage in stages:
        lines.append(
            f"    // pipeline stage {stage.index}: {stage.description} ({', '.join(stage.op_kinds)})"
        )
    for op in ir.ops:
        lines.extend(_emit_op(op))
    lines.extend(["    return", "  }", "}"])
    return "\n".join(lines)


def _emit_op(op: Operation) -> list[str]:
    attrs = _format_attrs(op.attrs)
    operands = ", ".join(_value_name(operand) for operand in op.operands)
    result = f"{_value_name(op.result)} = " if op.result is not None else ""
    if operands:
        body = f"{result}tileflow.{op.kind} {operands}{attrs}"
    else:
        body = f"{result}tileflow.{op.kind}{attrs}"
    return [f"    {body}"]


def _value_name(value: Value | None) -> str:
    if value is None:
        return ""
    return value.name


def _format_attrs(attrs: dict[str, object]) -> str:
    if not attrs:
        return ""
    items = ", ".join(f"{key} = {value!r}" for key, value in attrs.items())
    return f" {{{items}}}"

