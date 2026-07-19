import pytest

# pytest.importorskip("mlir.ir")

from tileflow.compiler.mlir_emitter import emit_upstream_mlir
from tileflow.language.ir import BufferType, FloatType, IRBuilder, OpName, Region


def _build_scalar_kernel():
    f32 = FloatType(32)
    buffer_type = BufferType((16,), f32)
    with IRBuilder("scalar_add") as builder:
        source = builder.argument("source", 0, buffer_type)
        output = builder.output("output", 0, buffer_type)
        index = builder.const(0)
        loaded = builder.load(source, [index], type_=f32)
        increment = builder.const(1.0, f32)
        result = builder.binary(OpName.ADD, loaded, increment)
        builder.store(result, output, [index])
        builder.return_op([output])
    return builder.ir


def _build_serial_loop_kernel():
    f32 = FloatType(32)
    buffer_type = BufferType((16,), f32)
    with IRBuilder("serial_copy") as builder:
        source = builder.argument("source", 0, buffer_type)
        output = builder.output("output", 0, buffer_type)
        lower = builder.const(0)
        upper = builder.const(16)
        step = builder.const(1)
        body = Region()
        iv = builder.new_value(lower.type, owner=body.entry)
        body.entry.args.append(iv)
        with builder.region(body):
            loaded = builder.load(source, [iv], type_=f32)
            builder.store(loaded, output, [iv])
        builder.append_op(
            OpName.SERIAL_FOR,
            [lower, upper, step],
            attrs={"rank": 1},
            regions=[body],
        )
        builder.return_op([output])
    return builder.ir


def _build_if_kernel():
    f32 = FloatType(32)
    buffer_type = BufferType((1,), f32)
    with IRBuilder("conditional_store") as builder:
        output = builder.output("output", 0, buffer_type)
        index = builder.const(0)
        value = builder.const(2.0, f32)
        zero = builder.const(0.0, f32)
        condition = builder.compare(OpName.GT, value, zero)
        then_region = Region()
        with builder.region(then_region):
            builder.store(value, output, [index])
        builder.if_op(condition, then_region)
        builder.return_op([output])
    return builder.ir


def test_emits_scalar_arithmetic_and_memory_operations():
    module = emit_upstream_mlir(_build_scalar_kernel())
    assert module.operation.verify()
    text = str(module)
    assert "arith.addf" in text
    assert "memref.load" in text
    assert "memref.store" in text


def test_emits_serial_loop_as_scf_for():
    module = emit_upstream_mlir(_build_serial_loop_kernel())
    assert module.operation.verify()
    assert "scf.for" in str(module)


def test_emits_comparison_and_structured_if():
    module = emit_upstream_mlir(_build_if_kernel())
    assert module.operation.verify()
    text = str(module)
    assert "arith.cmpf ogt" in text
    assert "scf.if" in text
