@tilelang.jit
def tl_copy_1d_serial(A):
    # The host/declaration part of TileLang script.
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    # The body of the kernel function is written in TileLang DSL.
    # We use T.Kernel to launch a kernel.
    with T.Kernel(1, threads=1) as _:
        # Here T.copy is a built-in TileOp in TileLang.
        # It will automatically utilize available threads in the block
        # to do efficient memory copy (including auto parallelism and vectorization)
        # As we only launch one thread here, it will be lowered into a serial loop copy
        # with certain bit width vectorization (like 128 bits per copy).
        T.copy(A, B)

    return B


@tilelang.jit
def tl_copy_1d_multi_threads(A):
    # The host/declaration part of TileLang script.
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    with T.Kernel(1, threads=256) as _:
        # Here T.copy is a built-in TileOp in TileLang.
        # It will automatically utilize available threads in the block
        # to do efficient memory copy (including auto parallelism and vectorization)
        # As we only launch one thread here, it will be lowered into a serial loop copy
        # with certain bit width vectorization (like 128 bits per copy).
        T.copy(A, B)

    return B


def run_copy_1d_multi_threads():
    print("\n=== Copy 1D Multi-threads ===\n")
    N = 1024 * 256

    test_puzzle(tl_copy_1d_multi_threads, ref_copy_1d, {"N": N})

    # This may take a while since N is large
    bench_puzzle(
        tl_copy_1d_serial,
        ref_copy_1d,
        {"N": N},
        bench_name="TL Serial",
        bench_torch=True,
    )
    bench_puzzle(
        tl_copy_1d_multi_threads,
        ref_copy_1d,
        {"N": N},
        bench_name="TL Multi-threads",
        bench_torch=False,
    )


"""
Finally, we want to parallelize the copy operation across multiple blocks.
We use BLOCK_N to represent the number of elements each block should copy.
The rest of the implementation is similar to the previous version. We assume that N is divisible
by BLOCK_N.

Note: You will need to handle the memory access ranges for different blocks. Fortunately,
we have `bx` (the block index) available, so you can compute the start and end indices for
each block accordingly.
"""


@tilelang.jit
def tl_copy_1d_parallel(A, BLOCK_N: int):
    # The host/declaration part of TileLang script.
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    num_threads_per_block = 256
    num_per_thread = T.ceildiv(BLOCK_N, num_threads_per_block)
    # TODO: Implement this function
    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=num_threads_per_block) as bx:
        # elements that need to be taken care of for each block:
        # N = BLOCK_N * num_threads_per_block * num_per_thread
        # for thread_idx, item_idx in T.Parallel(num_threads_per_block, num_per_thread):
        #     idx = bx * BLOCK_N + thread_idx * num_per_thread + item_idx
        #     B[idx] = A[idx]
        # or:
        T.copy(
            A[bx * BLOCK_N : (bx + 1) * BLOCK_N],
            B[bx * BLOCK_N : (bx + 1) * BLOCK_N],
        )
    return B


def run_copy_1d_parallel():
    print("\n=== Copy 1D Parallel ===\n")
    N = 1024 * 256
    BLOCK_N = 1024
    test_puzzle(tl_copy_1d_parallel, ref_copy_1d, {"N": N, "BLOCK_N": BLOCK_N})
    bench_puzzle(
        tl_copy_1d_parallel,
        ref_copy_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL Parallel",
        bench_torch=True,
    )


if __name__ == "__main__":
    # run_copy_1d_serial()
    # run_copy_1d_multi_threads()
    run_copy_1d_parallel()
