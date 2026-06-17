# Step 5.2: Dialect Operation Spec

Scope: 2-3 days.

Goal: design the initial TileFlow dialect on paper before writing C++.

Build:

- Add `docs/dialect.md`.
- Specify initial ops: kernel, program_id, load, store, alloc, layout_map, pipeline ops.
- Specify traits, side effects, attributes, and result types.
- Write example MLIR manually.

Tests:

- Review examples for consistency.
- Every op has at least one valid and one invalid example.

References:

- `ref/Enigma-DSL/Enigma-Dialect/README.md`

Learning focus:

- Dialect design.
- Separating source DSL concepts from IR concepts.

Writing output:

- Blog seed: "Designing an MLIR dialect before writing C++."

