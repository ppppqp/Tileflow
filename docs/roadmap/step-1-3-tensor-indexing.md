# Step 1.3: Tensor Indexing and Memory Ops

Scope: 2-3 days.

Goal: represent logical tensor indexing independently from physical layout.

Build:

- Support one-dimensional and multi-dimensional indexing.
- Represent load/store index tuples explicitly.
- Add memory spaces: global, shared, local.
- Introduce bounds as metadata, not runtime checks.

Tests:

- `a[i]`, `a[i, j]`, and `c[i, j] = value`.
- Rank mismatch diagnostics.
- Memory-space metadata in IR.

References:

- `../tilelang/tilelang/language/allocate.py`
- `../tilelang/tilelang/layout/layout.py`

Learning focus:

- Tensor semantics.
- Logical index spaces.
- Physical layout as a later decision.

Writing output:

- Blog seed: "A tensor access is not just a pointer offset."

