import tileflow
import tileflow.language as T


@tileflow.jit
def add(A, B, N: int):
    A: T.Tensor((N,), T.float32)
    B: T.Tensor((N,), T.float32)
    C = T.Tensor((N,), T.float32)

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
