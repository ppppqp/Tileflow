# Step 6.2: Python FFI Smoke Binding

Scope: 2-3 days.

Goal: expose one native MLIR-related function to Python.

Build:

- Expose one native function: register dialect, parse module, or report version.
- Package it as an optional extension.
- Add import tests.

Tests:

- Import binding.
- Call native function.
- Skip cleanly if extension is not built.

References:

- `ref/Enigma-DSL/Enigma-Dialect/python/EnigmaModule.cpp`
- `../tilelang/tilelang/layout/layout.py`

Learning focus:

- C++/Python extension boundaries.
- ABI and packaging pain points.

Writing output:

- Blog seed: "The first FFI bridge in a compiler project."

