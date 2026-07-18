import logging

import tileflow
import tileflow.language as T

logging.basicConfig(level=logging.DEBUG)


@tileflow.jit
def reduce_sum(A, B, M: int, N: int):
    A: T.Tensor((M, N), T.float32)
    O = T.empty((N), T.float32)

    with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
        for tx in T.Parallel(128):
            i = bx * 128 + tx
            acc = 0.0
            for j in T.Serial(M):
                acc += A[j, i]
            O[i] = acc

    return O


def main():
    compiled = reduce_sum.compile(M=1024, N=1024)

    logging.debug("Compiled IR:\n%s", compiled.ir)
    logging.debug("Compiled MLIR:\n%s", compiled.mlir)


if __name__ == "__main__":
    main()
