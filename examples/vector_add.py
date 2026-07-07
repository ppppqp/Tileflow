import tileflow
import tileflow.language as T


@tileflow.jit
def vector_add(A, B, N: int):
    A: T.Tensor((N,), T.float32)
    B: T.Tensor((N,), T.float32)
    C = T.empty((N,), T.float32)

    with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
        for tx in T.Parallel(128):
            i = bx * 128 + tx
            C[i] = A[i] + B[i]

    return C


if __name__ == "__main__":
    compiled = vector_add.compile(N=1024)
    print(compiled.mlir)
