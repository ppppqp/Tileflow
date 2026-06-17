# Step 7.1: Lower to Standard MLIR Dialects

Scope: 2-3 days.

Goal: convert TileFlow ops into upstream MLIR dialects.

Build:

- Lower TileFlow ops to `func`, `scf`, `arith`, `memref`, and possibly `gpu`.
- Add conversion patterns.
- Add native pass pipeline.

Tests:

- `tileflow-opt --convert-tileflow-to-standard`.
- FileCheck output uses standard dialects.

References:

- MLIR conversion pattern docs.
- Enigma dialect pass structure.

Learning focus:

- Conversion patterns.
- Type conversion.
- Legality.

Writing output:

- Blog seed: "Lowering a custom dialect into standard MLIR."

