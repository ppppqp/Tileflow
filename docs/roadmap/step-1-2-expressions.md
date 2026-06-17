# Step 1.2: Expressions and Primitive Ops

Scope: 2-3 days.

Goal: trace scalar expression trees into stable SSA-style operations.

Build:

- Add arithmetic: add, sub, mul, div, mod, neg.
- Add comparisons and boolean operations.
- Add dtype propagation rules.
- Add constants with stable type inference.

Tests:

- Arithmetic op trace order.
- Constant policy.
- Type mismatch diagnostics.

References:

- `ref/Enigma-DSL/enigma/_tracing.py`

Learning focus:

- Operator overloading limits.
- SSA value modeling.
- Early type checking.

Writing output:

- Blog seed: "How Python operator overloading becomes compiler IR."

