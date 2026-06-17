# Step 5.3: Native Dialect Skeleton

Scope: 2-3 days.

Goal: create a minimal native MLIR project that can parse and print one TileFlow op.

Build:

- Create `mlir/` or `dialect/` native subproject.
- Add CMake, TableGen files, dialect registration, and one op.
- Build `tileflow-opt`.
- Add lit test for parse/print roundtrip.

Tests:

- `ninja check-tileflow`
- Parse/print test for `tileflow.kernel`.

References:

- `ref/Enigma-DSL/Enigma-Dialect/CMakeLists.txt`
- `ref/Enigma-DSL/Enigma-Dialect/test/lit.cfg.py`

Learning focus:

- MLIR build system.
- ODS/TableGen.
- lit and FileCheck.

Writing output:

- Blog seed: "My first MLIR dialect: CMake, TableGen, and one op."

