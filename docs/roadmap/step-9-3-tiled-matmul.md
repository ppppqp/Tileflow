# Step 9.3: Tiled Matmul Skeleton

Scope: 2-3 days.

Goal: represent a tiled matmul without requiring peak performance.

Build:

- Add a minimal tiled matmul DSL example.
- Represent shared memory tiles and pipeline stages.
- Add diagnostics for incompatible tile shapes.

Tests:

- Trace and pipeline shape are correct.
- Diagnostics catch incompatible tile shapes.

References:

- `../tilelang/tilelang/carver/template/matmul.py`

Learning focus:

- Tiling.
- Accumulators.
- Shared memory staging.

Writing output:

- Course chapter: "Matmul as a compiler design test."

