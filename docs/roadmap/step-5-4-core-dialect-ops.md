# Step 5.4: Add Core Dialect Ops

Scope: 2-3 days.

Goal: add enough native ops to represent vector add without fake syntax.

Build:

- Add program ID, load, store, return, and simple arithmetic ops, or intentionally reuse `arith`.
- Implement verifiers where useful.
- Add parse/print tests.

Tests:

- Valid examples parse.
- Invalid examples fail with expected diagnostics.

References:

- `ref/Enigma-DSL/Enigma-Dialect/README.md`
- MLIR upstream dialect docs.

Learning focus:

- Operation traits.
- Custom verification.
- Reusing standard MLIR dialects.

Writing output:

- Blog seed: "When to create a custom op and when to reuse MLIR."

