import tileflow
import tileflow.language as T
from tileflow.language.ir import OpName


@tileflow.jit
def reduce_sum(A, B, M: int, N: int):
    A: T.Tensor((M, N), T.float32)
    output = T.empty((N), T.float32)

    with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
        for tx in T.Parallel(128):
            i = bx * 128 + tx
            acc = 0.0
            for j in T.Serial(M):
                acc += A[j, i]
            output[i] = acc

    return output


@tileflow.jit
def pipelined_sum(A, K: int):
    A: T.Tensor((K,), T.float32)
    output = T.empty((1,), T.float32)
    with T.Kernel(1, threads=32):
        acc = 0.0
        for k in T.Pipelined(K, num_stages=3):
            acc += A[k]
        output[0] = acc
    return output


def test_compile_emits_mlir_and_pass_metadata():
    # Real native behavior is covered by mlir/FileCheck tests once tileflow-opt is built.
    compiled = reduce_sum.compile(M=1024, N=1024)
    assert compiled.name == "reduce_sum"
    assert "func.func @reduce_sum" in compiled.mlir
    kernel = next(op for op in compiled.ir.body.entry.ops if op.name == OpName.KERNEL)
    parallel = next(op for op in kernel.regions[0].entry.ops if op.name == OpName.PARALLEL)
    serial = next(op for op in parallel.regions[0].entry.ops if op.name == OpName.SERIAL_FOR)
    assert parallel.attrs["rank"] == 1
    assert serial.attrs["rank"] == 1


def test_pipelined_loop_has_a_distinct_operation():
    compiled = pipelined_sum.compile(K=32)
    kernel = next(op for op in compiled.ir.body.entry.ops if op.name == OpName.KERNEL)
    pipeline = next(
        op for op in kernel.regions[0].entry.ops if op.name == OpName.PIPELINED_FOR
    )
    assert pipeline.attrs == {"rank": 1, "num_stages": 3}
