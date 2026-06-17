# Step 3.3: Canonicalization and Simplification

Scope: 2-3 days.

Goal: add the first optimization pass while preserving side-effect correctness.

Build:

- Fold simple constants.
- Remove dead operations.
- Normalize index expressions.
- Add deterministic IR printing.

Tests:

- `i + 0` simplifies.
- Unused load is removed if side-effect free.
- Stores are never removed by accident.

References:

- MLIR canonicalization concepts.
- Existing TileFlow IR and verifier.

Learning focus:

- Side effects.
- Canonical forms.
- Why optimization starts small.

Writing output:

- Blog seed: "The first optimization pass in a tiny compiler."

