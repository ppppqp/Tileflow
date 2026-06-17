# Step 8.1: Compiled Artifact Model

Scope: 2-3 days.

Goal: make compilation outputs explicit, serializable, and cacheable.

Build:

- Define `CompiledKernel` fields for IR, MLIR, target code, binary path, params, and diagnostics.
- Add cache keys based on source, options, and target.
- Add export methods.

Tests:

- Cache key changes when source/options change.
- Export writes expected files.

References:

- `../tilelang/tilelang/engine/param.py`
- `../tilelang/tilelang/cache/`

Learning focus:

- Compiler products.
- Reproducibility and caching.

Writing output:

- Blog seed: "What does a compiler return?"

