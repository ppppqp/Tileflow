# Step 7.2: CPU Interpreter or LLVM Path

Scope: 2-3 days.

Goal: run a TileFlow-compiled kernel somewhere before chasing GPU performance.

Build:

- Pick the simplest executable path: interpreter, C emitter, or LLVM lowering.
- Run vector add on CPU first.
- Return results to Python.

Tests:

- Vector add correctness.
- Scalar argument correctness.
- Runtime argument mismatch errors.

References:

- `../tilelang/tilelang/engine/lower.py`
- MLIR execution engine examples if using LLVM.

Learning focus:

- Execution before GPU.
- Runtime ABI.

Writing output:

- Blog seed: "Making the compiler run something, even on CPU."

