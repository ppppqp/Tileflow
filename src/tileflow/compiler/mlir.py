"""Textual MLIR emitter for the bootstrap IR."""

from __future__ import annotations

from tileflow.ir import KernelIR, Operation, Value


def emit_mlir(ir: KernelIR) -> str:
    args = ", ".join(f"%{arg.name}: memref<*x{_mlir_type(arg.dtype)}>" for arg in ir.args)
    lines = [
        "module {",
        f"  func.func @{ir.name}({args}) {{",
    ]
    for op in ir.ops:
        lines.extend(_emit_op(op))
    lines.extend(["    return", "  }", "}"])
    return "\n".join(lines)


def _emit_op(op: Operation) -> list[str]:
    attrs = _format_attrs(op.attrs)
    operands = ", ".join(_value_name(operand) for operand in op.operands)
    result = f"{_value_name(op.result)} = " if op.result is not None else ""
    op_name = f'"tileflow.{op.kind}"'
    if operands:
        body = f"{result}{op_name}({operands}){attrs} : () -> ()"
    else:
        body = f"{result}{op_name}(){attrs} : () -> ()"
    return [f"    {body}"]


def _value_name(value: Value | None) -> str:
    if value is None:
        return ""
    return value.name


def _format_attrs(attrs: dict[str, object]) -> str:
    if not attrs:
        return ""
    items = ", ".join(f"{key} = {_format_attr_value(value)}" for key, value in attrs.items())
    return f" {{{items}}}"


def _format_attr_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_attr_value(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _mlir_type(dtype: str) -> str:
    return {
        "T.float16": "f16",
        "T.float32": "f32",
        "T.float64": "f64",
        "T.int8": "i8",
        "T.int16": "i16",
        "T.int32": "i32",
        "T.int64": "i64",
        "T.uint8": "i8",
        "T.uint16": "i16",
        "T.uint32": "i32",
        "T.uint64": "i64",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "unknown": "f32",
    }.get(str(dtype), str(dtype).removeprefix("T."))
