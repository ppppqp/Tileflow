# Step 3.2: Verification Passes

Scope: 2-3 days.

Goal: make IR invariants executable and catch mistakes close to their source.

Build:

- Add IR verifier: values defined before use, valid tensor args, valid attrs.
- Add type verifier.
- Add layout verifier.
- Run verifier before and after major passes.

Tests:

- Manually constructed invalid IR cases.
- User-facing DSL errors remain readable.
- Verifier points to the failing operation.

References:

- MLIR verifier concepts.
- `../tilelang/tilelang/engine/semantic_check.py`

Learning focus:

- Compiler invariants.
- Diagnostic quality.
- Fail-fast pipeline design.

Writing output:

- Blog seed: "Compiler verification is where good errors come from."

