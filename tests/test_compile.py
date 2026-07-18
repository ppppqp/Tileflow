import tileflow
import tileflow.language as T
from tileflow.language.ir import OpName, TensorType


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


# def test_compile_emits_mlir_and_pass_metadata():
#     compiled = add.compile(N=1024)

#     assert compiled.name == "add"


def test_compile_can_require_native_mlir():
    A = T.Tensor((1024,), T.float32)
    B = T.Tensor((1024,), T.float32)
    compiled = add.compile(A=A, B=B, N=1024)
    print(compiled.mlir)
    allocation = next(op for op in compiled.ir.body.entry.ops if op.name == OpName.ALLOC)
    assert isinstance(allocation.results[0].type, TensorType)
    assert allocation.results[0].type.memory_space == "global"
