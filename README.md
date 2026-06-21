# TileFlow

TileFlow is an experimental compiler for TileLang-compatible Python DSL kernels. The goal is to keep the source language close to TileLang while replacing the TVM/TIR backend path with a frontend IR designed for MLIR lowering.

The current goal is to parse TileLang-style Python functions into MLIR text and immediately delegate verification and optimization to native MLIR passes. Backend-specific lowering and runtime dispatch are future milestones.

## Direction

TileFlow is designed around this pipeline:

```text
Python @tileflow.jit + tileflow.language as T
    -> TileLang-compatible AST frontend
    -> TileFlow frontend IR
    -> layout inference
    -> pipelining analysis
    -> MLIR module text
    -> native MLIR dialect/lowering backend
```

The project is informed by two local references:

- `ref/Enigma-DSL`: Python DSL tracing, MLIR emission boundary, and a split between Python frontend and native dialect.
- `../tilelang`: tile-level programming model, layout abstractions, and pass pipeline organization.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python3 examples/vector_add.py
python3 examples/tilelang_matmul.py
python3 -m pytest -q
```

The examples require either the `tileflow_mlir` Python extension or a built
`tileflow-opt` available through `TILEFLOW_OPT` or `PATH`.

Example:

```python
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

compiled = add.compile(N=1024)
print(compiled.mlir)
```

## Repository Layout

```text
mlir/          Native MLIR dialect, passes, tools, and optional Python extension.
src/tileflow/
  dsl/          User-facing JIT decorator.
  language.py   TileLang-compatible `T` namespace.
  ir.py         Minimal SSA-style traced IR.
  layout.py     Layout model and inference helpers.
  compiler/     AST frontend, compile orchestration, native pass bridge, and MLIR emitter.
examples/       Small runnable DSL examples.
tests/          Smoke tests for tracing and pass behavior.
docs/           Design notes.
```

Native build notes: [docs/native-build.md](docs/native-build.md).

## Milestones

1. Python frontend
   - Parse TileLang-compatible tensor declarations, allocations, kernel launches, loops, copies, and stores.
   - Expand compatibility across common TileLang examples.
   - Keep parser/IR internals MLIR-oriented rather than TVM/TIR-shaped.

2. Layout inference
   - Move layout inference into native MLIR passes.
   - Propagate layouts through views, tiles, and shared-memory buffers.
   - Surface diagnostics when layouts are ambiguous or incompatible.

3. Pipelining
   - Model producer/consumer stages.
   - Represent async copy, wait, compute, and commit groups.
   - Lower pipeline annotations to MLIR attributes and ops.

4. MLIR integration
   - Replace text-only emission with MLIR Python bindings.
   - Add a native `tileflow` dialect for tensors, layouts, and pipeline stages.
   - Lower to target dialects such as GPU, NVVM/ROCDL, LLVM, or target-specific dialects.

## Current Status

This is a bootstrap, not a production compiler. It is useful for iterating on the frontend API, pass contracts, and emitted IR shape while the native MLIR dialect is designed.
