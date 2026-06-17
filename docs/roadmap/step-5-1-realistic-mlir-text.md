# Step 5.1: Realistic MLIR Text Emission

Scope: 2-3 days.

Goal: emit MLIR text that is close enough to real syntax to guide native dialect work.

Build:

- Emit types, attributes, and regions in a realistic MLIR style.
- Separate generic MLIR emission from TileFlow dialect op names.
- Add FileCheck-style expected output files even before native tools exist.

Tests:

- Golden files under `tests/mlir/`.
- Stable output after canonicalization.

References:

- `ref/Enigma-DSL/enigma/compiler/mlir_emitter.py`
- `ref/Enigma-DSL/Enigma-Dialect/test/`

Learning focus:

- MLIR syntax.
- Regions, attributes, types, and assembly forms.

Writing output:

- Blog seed: "Learning MLIR by emitting text first."

