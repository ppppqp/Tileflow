# Step 2.4: Shared Memory and Tile Views

Scope: 2-3 days.

Goal: model tiled movement through memory hierarchy before async pipelining.

Build:

- Add `tf.alloc_shared(shape, dtype, layout=...)`.
- Add tile views over global tensors.
- Model copy from global tile to shared tile.
- Keep async behavior for Phase 4.

Tests:

- Shared allocation appears in IR.
- Copy op carries source and destination layouts.
- Rank and shape mismatch diagnostics.

References:

- `../tilelang/tilelang/language/allocate.py`
- `../tilelang/tilelang/language/copy_op.py`

Learning focus:

- Memory hierarchy modeling.
- Tiling before backend lowering.

Writing output:

- Blog seed: "Introducing shared memory without writing a backend yet."

