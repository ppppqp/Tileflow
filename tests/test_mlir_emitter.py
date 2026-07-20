from tileflow.compiler.mlir_emitter import create_mlir_context, emit_upstream_mlir
from tileflow.language.ir import BufferType, FloatType, IRBuilder, IndexType, OpName, Region


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


def _build_parallel_kernel():
    f32 = FloatType(32)
    buffer_type = BufferType((16,), f32)
    with IRBuilder("parallel_copy") as builder:
        source = builder.argument("source", 0, buffer_type)
        output = builder.output("output", 0, buffer_type)
        grid = builder.const(1)
        kernel_body = Region()
        program_id = builder.new_value(IndexType(), owner=kernel_body.entry)
        kernel_body.entry.args.append(program_id)
        with builder.region(kernel_body):
            lower = builder.const(0)
            upper = builder.const(16)
            step = builder.const(1)
            parallel_body = Region()
            index = builder.new_value(IndexType(), owner=parallel_body.entry)
            parallel_body.entry.args.append(index)
            with builder.region(parallel_body):
                value = builder.load(source, [index], type_=f32)
                builder.store(value, output, [index])
            builder.append_op(
                OpName.PARALLEL,
                [lower, upper, step],
                attrs={"rank": 1},
                regions=[parallel_body],
            )
        builder.append_op(
            OpName.KERNEL,
            [grid],
            attrs={"rank": 1, "threads": 128},
            regions=[kernel_body],
        )
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


def test_emits_tileflow_kernel_and_parallel_regions():
    module = emit_upstream_mlir(_build_parallel_kernel())
    assert module.operation.verify()
    text = str(module)
    assert '"tileflow.kernel"' in text
    assert '"tileflow.parallel"' in text
    assert "operandSegmentSizes = array<i32: 1, 1, 1>" in text


def test_native_tileflow_dialect_is_registered():
    context = create_mlir_context()
    assert context.is_registered_operation("tileflow.kernel")
    assert context.is_registered_operation("tileflow.parallel")
