# TileFlow

TileFlow is an experimental Python DSL compiler for tiled GPU kernels. The goal is a TileLang-like authoring experience with first-class layout inference and an explicit lowering path to MLIR.

The initial scaffold is intentionally small: it captures Python DSL operations into a lightweight IR, runs layout inference and pipeline planning passes, and emits inspectable MLIR-like text. Native MLIR dialect bindings, backend-specific lowering, and runtime dispatch are future milestones.

## Direction

TileFlow is designed around this pipeline:

```text
Python @tileflow.kernel
    -> traced TileFlow IR
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
python examples/vector_add.py
pytest
```

Example:

```python
import tileflow as tf

@tf.kernel
def add(a: tf.TensorType("f32"), b: tf.TensorType("f32"), c: tf.TensorType("f32")):
    i = tf.program_id(0)
    c[i] = a[i] + b[i]

compiled = tf.compile(add)
print(compiled.mlir)
```

## Repository Layout

```text
src/tileflow/
  dsl/          User-facing decorators, tensor proxies, and types.
  ir.py         Minimal SSA-style traced IR.
  layout.py     Layout model and inference helpers.
  compiler/     Compile orchestration, passes, and MLIR emitter.
examples/       Small runnable DSL examples.
tests/          Smoke tests for tracing and pass behavior.
docs/           Design notes.
```

## Milestones

1. Python frontend
   - Trace tensor indexing, arithmetic, stores, constants, and program IDs.
   - Add structured control flow capture.
   - Define ergonomic TileLang-inspired allocation/copy/reduction syntax.

2. Layout inference
   - Infer default layouts from tensor rank and access patterns.
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

