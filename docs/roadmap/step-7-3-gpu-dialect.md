# Step 7.3: GPU Dialect Lowering Sketch

Scope: 2-3 days.

Goal: map TileFlow kernel concepts onto MLIR's GPU dialect.

Build:

- Map kernel launch concepts to `gpu.module` and `gpu.func`.
- Represent block/thread IDs.
- Lower memory spaces to address spaces.
- Keep performance out of scope.

Tests:

- FileCheck for `gpu.module`, `gpu.func`, and ID ops.

References:

- MLIR GPU dialect docs.
- `../tilelang/tilelang/backend/target.py`

Learning focus:

- MLIR GPU dialect.
- Target-independent GPU representation.

Writing output:

- Blog seed: "A small Python DSL meets MLIR's GPU dialect."

