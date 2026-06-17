# Step 4.3: Async Copy Semantics

Scope: 2-3 days.

Goal: define async copy as a semantic contract before choosing target instructions.

Build:

- Add async copy operation to the DSL.
- Track copy groups and waits.
- Add target capability flags.
- Make unsupported async copy fall back or error explicitly.

Tests:

- Async copy produces group metadata.
- Missing wait before use is diagnosed.
- Unsupported target capability errors.

References:

- `../tilelang/tilelang/language/copy_op.py`
- `ref/Enigma-DSL/tests/portable/test_async_copy.py`

Learning focus:

- Memory ordering.
- Semantic checks before backend lowering.

Writing output:

- Blog seed: "Async copy is a contract, not just an instruction."

