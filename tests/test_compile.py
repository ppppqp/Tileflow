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


def test_compile_can_require_native_mlir():
    A = T.Tensor((1024,), T.float32)
    B = T.Tensor((1024,), T.float32)
    compiled = add.compile(A=A, B=B, N=1024)
    print(compiled.mlir)
    assert not any(op.name == OpName.ALLOC for op in compiled.ir.body.entry.ops)
    assert len(compiled.ir.outputs) == 1
    output = compiled.ir.outputs[0]
    assert output.source_name == "C"
    assert isinstance(output.value.type, TensorType)
    assert output.value.type.memory_space == "global"
