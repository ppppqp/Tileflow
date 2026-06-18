import logging

import tileflow
import tileflow.language as T


@tileflow.jit
def add(A, B, N: int):
    A: T.Tensor((N,), T.float32)
    B: T.Tensor((N,), T.float32)
    C = T.empty((N,), T.float32)

    with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
        for tx in T.Parallel(128):
            i = bx * 128 + tx
            C[i] = A[i] + B[i]

    return C


def test_compile_emits_mlir_and_pass_metadata():
    compiled = add.compile(N=1024)

    assert compiled.name == "add"
    assert "func.func @add" in compiled.mlir
    assert "tileflow.tensor_decl" in compiled.mlir
    assert "tileflow.kernel_launch" in compiled.mlir
    assert "tileflow.store" in compiled.mlir
    assert set(compiled.layouts) == {"A", "B", "C"}
    assert "tileflow.parallel_for" in compiled.mlir
