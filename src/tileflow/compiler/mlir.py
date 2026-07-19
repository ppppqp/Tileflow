"""Textual MLIR emitter for the bootstrap IR."""

from __future__ import annotations

from tileflow.language.ir import KernelIR, Operation, Type, Value


def emit_mlir(ir: KernelIR) -> str:
    signature = [*ir.params, *ir.outputs]
    args = ", ".join(
        f"{item.value.ir_name}: {_mlir_value_type(item.value.type)}" for item in signature
    )
    lines = [
        "module {",
        f"  func.func @{ir.name}({args}) {{",
    ]
    lines.extend(_emit_block(ir.body.entry, indent="    "))
    lines.extend(["    return", "  }", "}"])
    return "\n".join(lines)


def _emit_block(block, *, indent: str) -> list[str]:
    lines: list[str] = []
    for op in block.ops:
        lines.extend(_emit_op(op, indent=indent))
    if block.terminator is not None:
        lines.extend(_emit_op(block.terminator, indent=indent))
    return lines


def _emit_op(op: Operation, *, indent: str) -> list[str]:
    attrs = _format_attrs(op.attrs)
    operands = ", ".join(_value_name(operand) for operand in op.operands)
    result = ", ".join(_value_name(value) for value in op.results)
    result_prefix = f"{result} = " if result else ""
    op_name = str(op.name)
    if not op_name.startswith(("tileflow.", "arith.", "scf.", "memref.", "func.")):
        op_name = f"tileflow.{op_name}"
    op_name = f'"{op_name}"'
    if operands:
        body = f"{result_prefix}{op_name}({operands}){attrs} : () -> ()"
    else:
        body = f"{result_prefix}{op_name}(){attrs} : () -> ()"
    lines = [f"{indent}{body}"]
    for index, region in enumerate(op.regions):
        lines.append(f"{indent}// region {index}")
        lines.extend(_emit_block(region.entry, indent=indent + "  "))
    return lines


def _value_name(value: object) -> str:
    if isinstance(value, Value):
        return value.ir_name
    return str(value)


def _mlir_value_type(type_: Type) -> str:
    text = str(type_)
    if text.startswith("tensor<"):
        element = getattr(type_, "element_type", "f32")
        return f"memref<*x{_mlir_type(str(element))}>"
    return _mlir_type(text)


def _format_attrs(attrs: dict[str, object]) -> str:
    if not attrs:
        return ""
    items = ", ".join(f"{key} = {_format_attr_value(value)}" for key, value in attrs.items())
    return f" {{{items}}}"


def _format_attr_value(value: object) -> str:
    if isinstance(value, Value):
        return f'"{value.ir_name}"'
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
        "f16": "f16",
        "f32": "f32",
        "f64": "f64",
        "i1": "i1",
        "i8": "i8",
        "i16": "i16",
        "i32": "i32",
        "i64": "i64",
        "index": "index",
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
