# Native MLIR Build

TileFlow's Python frontend can run without LLVM, but native verification and optimization require building the MLIR scaffold.

## Prerequisites

You need an LLVM/MLIR build that exports `MLIRConfig.cmake` and `LLVMConfig.cmake`.

Typical environment variables:

```bash
export MLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
export LLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
```

If using Enigma's helper LLVM build as a reference, source its activation script before configuring TileFlow:

```bash
source ~/.local/enigma-llvm/activate.sh
```

## Configure and Build

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR="$MLIR_DIR" \
  -DLLVM_DIR="$LLVM_DIR"

ninja -C build tileflow-opt
```

If `pybind11` is available to CMake, the build also creates the optional `tileflow_mlir` Python extension.

## Python Integration

The Python compiler requires native execution and resolves it in this order:

1. Import `tileflow._mlir.tileflow_mlir.PassPipeline`.
2. Run `tileflow-opt` from `TILEFLOW_OPT` or `PATH`.

Useful environment variables:

```bash
export TILEFLOW_OPT=/absolute/path/to/build/mlir/tools/tileflow-opt/tileflow-opt
export TILEFLOW_MLIR_PIPELINE='tileflow-frontend-verify,canonicalize,cse'
```

Then run:

```bash
PYTHONPATH=src python3 examples/vector_add.py
```

## Editable Development

Use editable install for the pure Python package:

```bash
python -m pip install -e ".[dev]"
```

After building `tileflow_mlir`, copy the extension into the package namespace:

```bash
mkdir -p src/tileflow/_mlir
cp build/mlir/python/tileflow_mlir*.so src/tileflow/_mlir/
python -c "from tileflow._mlir.tileflow_mlir import PassPipeline; print(PassPipeline)"
```

This mirrors the wheel layout we want long term: native extension modules live
inside the `tileflow` package, not as unrelated top-level imports.

## Current Native Scope

The native scaffold currently includes:

- `tileflow` dialect ODS definitions for the frontend placeholder ops.
- `tileflow-opt` optimizer driver.
- `tileflow-frontend-verify` no-op pass hook.
- Optional `tileflow_mlir.PassPipeline` Python extension.
- A FileCheck-style smoke test at `tests/mlir/vector_add.mlir`.

There is no Python optimization fallback. If neither `tileflow_mlir` nor
`tileflow-opt` is available, `tileflow.jit(...).compile(...)` fails.

The next step is to move real semantics out of Python:

- typed op verifiers
- region-based loop/kernel ops
- layout inference
- pipeline planning
- lowering to upstream MLIR dialects
