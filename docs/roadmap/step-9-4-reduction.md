# Step 9.4: Reduction Kernel

Scope: 2-3 days.

Goal: add a kernel pattern that forces reduction semantics into the DSL and IR.

Build:

- Add row-sum or RMSNorm-style reduction.
- Add reduction op or pattern.
- Model local/shared memory accumulation.

Tests:

- Reduction trace contains explicit reduction structure.
- Layout inference recognizes read-many/write-one pattern.

References:

- `ref/Enigma-DSL/examples/benchmark_rmsnorm.py`
- TileLang reduce ops.

Learning focus:

- Associativity.
- Parallel reductions.

Writing output:

- Course chapter: "Reductions force the DSL to grow up."

