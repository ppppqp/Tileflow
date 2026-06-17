# Step 9.2: Matrix Transpose

Scope: 2-3 days.

Goal: use transpose as the first layout-sensitive kernel.

Build:

- Add transpose example.
- Demonstrate non-contiguous access and layout inference.
- Add shared memory tile variant if Phase 2.4 is complete.

Tests:

- Inference identifies strided access.
- Emitted layout metadata differs from vector add.

References:

- TileLang layout and copy examples.

Learning focus:

- Access patterns.
- Memory coalescing.

Writing output:

- Course chapter: "Transpose: the first layout-sensitive kernel."

