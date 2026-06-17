# Step 4.2: Double Buffering

Scope: 2-3 days.

Goal: model prologue, steady state, and epilogue stages for tiled loops.

Build:

- Add `num_stages` and buffer slot analysis.
- Rewrite tiled loop bodies into prologue, steady-state, and epilogue structure.
- Emit readable MLIR-like text showing pipeline slots.

Tests:

- Two-stage pipeline assigns alternating slots.
- Prologue and epilogue are emitted.
- Invalid stage count errors clearly.

References:

- `../tilelang/tilelang/cuda/pipeline.py`

Learning focus:

- Software pipelining.
- Latency hiding.
- Loop rewriting.

Writing output:

- Blog seed: "Double buffering from first principles."

