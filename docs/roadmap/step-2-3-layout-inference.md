# Step 2.3: First Layout Inference Pass

Scope: 2-3 days.

Goal: infer default layouts and explain every inference decision.

Build:

- Infer physical layouts for tensor args from rank and access pattern.
- Represent constraints and conflicts explicitly.
- Add user layout annotations that override inference.
- Emit layout attributes in MLIR-like output.

Tests:

- Inferred contiguous vector layout.
- Inferred row-major matrix layout.
- User annotation wins over default.
- Conflicting constraints produce diagnostics.

References:

- `../tilelang/tilelang/layout/layout.py`
- `../tilelang/tilelang/analysis/`

Learning focus:

- Constraint solving without overengineering.
- Explainable inference.

Writing output:

- Blog seed: "A tiny layout inference engine."

