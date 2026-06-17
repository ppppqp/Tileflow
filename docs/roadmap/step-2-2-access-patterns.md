# Step 2.2: Access Pattern Collection

Scope: 2-3 days.

Goal: collect facts about how tensors are read and written.

Build:

- Add an analysis pass that records every tensor access.
- Classify access patterns: contiguous, strided, broadcast, reduction-like.
- Attach source locations if possible.
- Produce human-readable diagnostics.

Tests:

- Vector add is contiguous.
- Matrix transpose is strided.
- Reduction reads many values and writes one value.

References:

- `../tilelang/tilelang/analysis/`
- `../tilelang/tilelang/carver/roller/shape_inference/`

Learning focus:

- Compiler analysis as facts plus confidence.
- Diagnostics as compiler output.

Writing output:

- Blog seed: "What the compiler can infer from indexing alone."

