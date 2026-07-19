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

Build the native MLIR tool first, then run the Python examples.

### Configure LLVM/MLIR

If your local LLVM build is at `/home/qiping-pan/Documents/workspace/llvm/llvm-project/build`:

```bash
export LLVM_BUILD=/home/qiping-pan/Documents/workspace/llvm/llvm-project/build
export LLVM_DIR="$LLVM_BUILD/lib/cmake/llvm"
export MLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
export MLIR_PYTHON_ROOT="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
export PYTHONPATH="$MLIR_PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$LLVM_BUILD/bin:$PATH"
```

If `mlir-tblgen` or `llvm-tblgen` is missing, build them in the LLVM tree:

```bash
ninja -C "$LLVM_BUILD" llvm-tblgen mlir-tblgen
```

### Configure TileFlow

For `tileflow-opt` only:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DTILEFLOW_ENABLE_PYTHON_EXTENSION=OFF
```

For the optional `tileflow_mlir` pybind extension:

```bash
python -m pip install pybind11

cmake -S . -B build -G Ninja \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DTILEFLOW_ENABLE_PYTHON_EXTENSION=ON \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
```

### Build

Generate TableGen `.inc` files:

```bash
ninja -C build TileFlowDialectIncGen TileFlowPassIncGen
```

Build the native optimizer:

```bash
ninja -C build tileflow-opt
```

Build the optional Python extension:

```bash
ninja -C build tileflow_mlir
```

The direct Python-to-MLIR emitter also requires LLVM's MLIR Python bindings.
Configure and build them in the LLVM checkout, then expose the generated package:

```bash
cmake -S "$LLVM_BUILD/../llvm" -B "$LLVM_BUILD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build "$LLVM_BUILD" --target MLIRPythonModules -j"$(nproc)"

export MLIR_PYTHON_ROOT="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
export PYTHONPATH="$MLIR_PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Add the two exports to `~/.bashrc` to make the bindings available in new shells.
Verify the setup with:

```bash
.venv/bin/python -c "from mlir import ir; print(ir.Context())"
```

For editable development, install the Python package and copy the extension into
the package namespace:

```bash
python -m pip install -e ".[dev]"
mkdir -p src/tileflow/_mlir
cp build/mlir/python/tileflow_mlir*.so src/tileflow/_mlir/
python -c "from tileflow._mlir.tileflow_mlir import PassPipeline; print(PassPipeline)"
```

### Run

Point Python at the native optimizer if you are not using the pybind extension:

```bash
export TILEFLOW_OPT="$PWD/build/mlir/tools/tileflow-opt/tileflow-opt"
```

```bash
python -m pip install -e ".[dev]"
python3 examples/vector_add.py
python3 examples/tilelang_matmul.py
python3 -m pytest -q
```

The examples require either the `tileflow._mlir.tileflow_mlir` Python extension
or a built `tileflow-opt` available through `TILEFLOW_OPT` or `PATH`.

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
