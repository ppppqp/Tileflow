# Step 6.1: MLIR Python Bindings Strategy

Scope: 2-3 days.

Goal: decide how Python talks to native MLIR during development and packaging.

Build:

- Decide whether to start with subprocess text roundtrip or Python bindings.
- Document local LLVM/MLIR setup.
- Add a script to run `tileflow-opt` on emitted MLIR.
- Keep the Python package usable without native dependencies.

Tests:

- Python emits MLIR text.
- Native tool parses it when available.
- Tests skip cleanly when native tools are absent.

References:

- `ref/Enigma-DSL/Enigma-Dialect/python/`
- `ref/Enigma-DSL/Enigma-Dialect/scripts/`

Learning focus:

- Optional native dependencies.
- Developer experience around MLIR.

Writing output:

- Blog seed: "Bridging Python and MLIR without making imports painful."

