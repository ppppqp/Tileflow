# Step 6.3: Replace Text Emitter Internals

Scope: 2-3 days.

Goal: build real MLIR modules from Python while preserving text output for debugging.

Build:

- Build MLIR modules through Python bindings or native helper APIs.
- Keep textual output as a debugging/export format.
- Add roundtrip tests against `tileflow-opt`.

Tests:

- Python IR to native MLIR module.
- Native parse/print roundtrip.
- Golden output remains stable enough for tests.

References:

- MLIR Python binding examples.
- `ref/Enigma-DSL/enigma/compiler/mlir_emitter.py`

Learning focus:

- MLIR object model.
- Contexts, locations, insertion points, and ownership.

Writing output:

- Blog seed: "From strings to real MLIR modules."

