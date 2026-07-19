"""Compile-smoke coverage adapted from the TileLang puzzle examples.

The copied puzzle files remain useful as reference implementations, but depend on
TileLang, PyTorch, and its benchmark harness.  These kernels isolate the frontend
constructs TileFlow can trace today.  Runtime correctness and benchmark checks can
be added beside each case once ``CompiledKernel`` is executable.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import tileflow
import tileflow.language as T
from tileflow.compiler.compiler import CompiledKernel
from tileflow.language.ir import OpName


@tileflow.jit
def copy_1d(A, N: int, block_n: int):
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)
    with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
        for tx in T.Parallel(block_n):
            i = bx * block_n + tx
            B[i] = A[i]
    return B


@tileflow.jit
def outer_add(A, B, N: int, M: int, block_n: int, block_m: int):
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((M,), T.float16)
    C = T.empty((N, M), T.float16)
    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (
        bx,
        by,
    ):
        for i, j in T.Parallel(block_n, block_m):
            ni = bx * block_n + i
            mj = by * block_m + j
            C[ni, mj] = A[ni] + B[mj]
    return C


@tileflow.jit
def mul_relu(A, B, N: int, M: int, block_n: int, block_m: int):
    A: T.Tensor((N, M), T.float16)
    B: T.Tensor((M,), T.float16)
    C = T.empty((N, M), T.float16)
    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (
        bx,
        by,
    ):
        for i, j in T.Parallel(block_n, block_m):
            ni = bx * block_n + i
            mj = by * block_m + j
            product = A[ni, mj] * B[mj]
            C[ni, mj] = product if product > 0 else 0
    return C


@tileflow.jit
def reduce_sum(A, N: int, M: int, block_n: int):
    A: T.Tensor((N, M), T.float32)
    B = T.empty((N,), T.float32)
    with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
        for tx in T.Parallel(block_n):
            row = bx * block_n + tx
            acc = 0.0
            for col in T.Serial(M):
                acc += A[row, col]
            B[row] = acc
    return B


@tileflow.jit
def scalar_attention(Q, K, V, B: int, S: int, block_b: int):
    Q: T.Tensor((B, S), T.float32)
    K: T.Tensor((B, S), T.float32)
    V: T.Tensor((B, S), T.float32)
    output = T.empty((B, S), T.float32)
    with T.Kernel(T.ceildiv(B, block_b), threads=128) as bx:
        for bi in T.Parallel(block_b):
            row = bx * block_b + bi
            denom = 0.0
            for col in T.Serial(S):
                denom += Q[row, col] * K[row, col]
            for col in T.Serial(S):
                output[row, col] = Q[row, col] * K[row, col] * V[row, col] / denom
    return output


@tileflow.jit
def matmul(A, B, M: int, N: int, K: int, block_m: int, block_n: int):
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(N, block_n), threads=128) as (
        bx,
        by,
    ):
        for i, j in T.Parallel(block_m, block_n):
            row = bx * block_m + i
            col = by * block_n + j
            acc = 0.0
            for k in T.Serial(K):
                acc += A[row, k] * B[k, col]
            C[row, col] = acc
    return C


@tileflow.jit
def conv1d(X, W, N: int, L: int, KL: int, block_l: int):
    X: T.Tensor((N, L), T.float16)
    W: T.Tensor((KL,), T.float16)
    output = T.empty((N, L), T.float16)
    with T.Kernel(N, T.ceildiv(L, block_l), threads=128) as (bx, by):
        for tx in T.Parallel(block_l):
            col = by * block_l + tx
            acc = 0.0
            for k in T.Serial(KL):
                acc += X[bx, col + k] * W[k]
            output[bx, col] = acc
    return output


CompileCase = tuple[str, object, dict[str, int]]

CASES: tuple[CompileCase, ...] = (
    ("copy", copy_1d, {"N": 256, "block_n": 64}),
    ("outer_add", outer_add, {"N": 64, "M": 32, "block_n": 8, "block_m": 8}),
    ("mul_relu", mul_relu, {"N": 64, "M": 32, "block_n": 8, "block_m": 8}),
    ("reduce_sum", reduce_sum, {"N": 64, "M": 32, "block_n": 8}),
    ("attention", scalar_attention, {"B": 16, "S": 32, "block_b": 4}),
    ("matmul", matmul, {"M": 32, "N": 32, "K": 16, "block_m": 8, "block_n": 8}),
    ("conv1d", conv1d, {"N": 4, "L": 32, "KL": 3, "block_l": 8}),
)


@pytest.mark.parametrize(("name", "kernel", "params"), CASES, ids=[case[0] for case in CASES])
def test_example_compiles(name: str, kernel: object, params: dict[str, int]):
    compiled: CompiledKernel = kernel.compile(**params)  # type: ignore[attr-defined]
    assert compiled.name
    assert "func.func" in compiled.mlir
    assert any(op.name == OpName.ALLOC for op in compiled.ir.body.entry.ops), name
    assert any(op.name == OpName.KERNEL for op in compiled.ir.body.entry.ops), name


# Future runtime tests can parameterize the same cases with two additional callables:
# a PyTorch reference function and an input factory.  Keep performance tests separately
# marked so normal CI does not include warmup or timing noise.
Reference = Callable[..., object]
