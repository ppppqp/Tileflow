import tileflow
import tileflow.language as T


@tileflow.jit
def reduce_sum(A, B, M: int, N: int):
    A: T.Tensor((M, N), T.float32)
    O = T.empty((N), T.float32)

    with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
        for tx in T.Parallel(128):
            i = bx * 128 + tx
            acc = 0.0
            for j in T.Sequential(M):
                acc += A[j, i]
            O[i] = acc

    return O


def test_compile_emits_mlir_and_pass_metadata():
    # Real native behavior is covered by mlir/FileCheck tests once tileflow-opt is built.
    compiled = reduce_sum.compile(M=1024, N=1024)
    assert compiled.name == "reduce_sum"
    assert "func.func @reduce_sum" in compiled.mlir
    assert "tileflow.parallel_for" in compiled.mlir
    assert "tileflow.sequential_for" in compiled.mlir
    assert "tileflow.end_sequential_for" in compiled.mlir
    assert "tileflow.end_parallel_for" in compiled.mlir
