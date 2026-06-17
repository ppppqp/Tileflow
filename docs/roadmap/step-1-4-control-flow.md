# Step 1.4: Structured Control Flow

Scope: 2-3 days.

Goal: capture loops and conditionals without losing compiler structure.

Build:

- Add `tf.range` and `tf.static_range`.
- Add `tf.if_then_else` or an AST preprocessing path for Python `if`.
- Decide whether the DSL uses tracing, AST parsing, or a hybrid.
- Represent loops and conditionals as nested IR regions, not flattened comments.

Tests:

- Simple counted loop.
- Static unrolled loop.
- Conditional store.

References:

- `ref/Enigma-DSL/enigma/compiler/preprocessor.py`
- `../tilelang/tilelang/language/loop.py`
- `../tilelang/tilelang/language/parser/`

Learning focus:

- Tracing versus AST rewriting.
- Python control-flow limits.
- Region-based IR.

Writing output:

- Blog seed: "Tracing, AST rewriting, and the control-flow cliff."

