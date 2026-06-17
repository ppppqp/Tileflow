# Step 2.1: Layout Algebra Core

Scope: 2-3 days.

Goal: represent layouts as functions from logical coordinates to physical coordinates or offsets.

Build:

- Replace the simple `Layout` with input shape, output shape, and index map.
- Add row-major, column-major, blocked, and swizzled constructors.
- Add `compose`, `inverse` where possible, `tile`, and `flatten`.
- Keep symbolic dimensions printable and testable.

Tests:

- Row-major and column-major offset calculations.
- Blocked layout mapping.
- Composition smoke tests.

References:

- `ref/Enigma-DSL/enigma/core.py`
- `../tilelang/tilelang/layout/layout.py`
- `../tilelang/tilelang/layout/swizzle.py`

Learning focus:

- Layout as a function.
- Physical memory order.
- Symbolic shape handling.

Writing output:

- Blog seed: "Layouts are functions, not metadata."

