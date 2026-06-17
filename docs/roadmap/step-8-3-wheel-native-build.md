# Step 8.3: Wheel and Optional Native Build

Scope: 2-3 days.

Goal: package TileFlow so pure Python work stays easy while native MLIR remains available.

Build:

- Keep pure Python install simple.
- Add optional native extension build path.
- Document local MLIR requirements.
- Decide whether native artifacts live in the same package or a separate package.

Tests:

- Pure Python import works without MLIR.
- Native tests run only when extension exists.

References:

- `ref/Enigma-DSL/README.md`
- `../tilelang/pyproject.toml`

Learning focus:

- Python packaging for compilers.
- Optional native dependencies.

Writing output:

- Blog seed: "Packaging a compiler that has both Python and C++."

