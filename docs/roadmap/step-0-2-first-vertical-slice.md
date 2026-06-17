# Step 0.2: Design the First Vertical Slice

Scope: 2-3 days.

Goal: define the smallest end-to-end TileFlow compiler story around vector add.

Build:

- Document the vector-add flow from Python DSL to traced IR to layout metadata to MLIR-like text.
- Freeze a tiny IR contract: args, values, ops, attributes, and diagnostics.
- Decide which concepts are target-neutral and which belong to backend lowering.
- Add `docs/ir.md` with example traced IR and emitted MLIR-like output.

Tests:

- Snapshot test for vector-add emitted text.
- Test that unsupported annotations fail with useful diagnostics.

References:

- `ref/Enigma-DSL/enigma/compiler/compiler.py`
- `ref/Enigma-DSL/enigma/compiler/mlir_emitter.py`

Learning focus:

- Compiler contracts.
- IR stability.
- End-to-end debugging.

Writing output:

- Blog seed: "The smallest useful IR for a Python kernel DSL."

