# Step 4.1: Pipeline IR

Scope: 2-3 days.

Goal: replace coarse pipeline comments with explicit pipeline IR.

Build:

- Add operations: `pipeline.begin`, `pipeline.copy`, `pipeline.wait`, `pipeline.compute`, `pipeline.commit`.
- Model dependencies between stages.
- Keep this target-neutral.
- Emit pipeline operations in MLIR-like output.

Tests:

- Load/compute/store vector add gets three stages.
- Tiled copy/compute kernel gets copy and compute stages.
- Dependency graph is acyclic.

References:

- `../tilelang/tilelang/cuda/pipeline.py`
- `../tilelang/tilelang/cpu/pipeline.py`

Learning focus:

- Scheduling versus lowering.
- Data dependency modeling.

Writing output:

- Blog seed: "Pipelining as an IR problem."

