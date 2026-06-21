# Native MLIR Architecture Direction

TileFlow should keep Python as a frontend and orchestration layer, while MLIR/LLVM owns IR verification, canonicalization, analysis, optimization, and lowering.

## Target Split

### Python

Python should own:

- TileLang-compatible source syntax: `@tileflow.jit`, `import tileflow.language as T`.
- AST parsing and compile-time parameter specialization.
- Construction of MLIR modules through either textual MLIR or MLIR Python bindings.
- User-facing compile API, diagnostics formatting, cache keys, and runtime artifact wrappers.
- Optional convenience wrappers for invoking native pass pipelines.

Python should not own long-term:

- Canonicalization.
- CSE/DCE.
- Layout verification.
- Pipeline planning and rewrites.
- Type conversion.
- Backend lowering.
- Target code emission.

### Native MLIR/C++

Native MLIR should own:

- The `tileflow` dialect operation definitions.
- Verifiers, traits, interfaces, and canonicalization patterns.
- Passes: layout inference, pipeline planning, async copy legalization, memory-space legalization, lowering to standard dialects, and backend-specific lowering.
- Tools: `tileflow-opt`, later `tileflow-translate` or backend runners.
- Python bindings for dialect registration and pass execution.

## Reference Findings

### Enigma

Enigma is the closest model for a small MLIR-based project.

Relevant files:

- `ref/Enigma-DSL/Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaOps.td`
- `ref/Enigma-DSL/Enigma-Dialect/include/enigma/Dialect/Enigma/IR/CMakeLists.txt`
- `ref/Enigma-DSL/Enigma-Dialect/python/CMakeLists.txt`
- `ref/Enigma-DSL/Enigma-Dialect/python/mlir_enigma/dialects/EnigmaOps.td`
- `ref/Enigma-DSL/Enigma-Dialect/python/mlir_enigma/dialects/enigma.py`
- `ref/Enigma-DSL/Enigma-Dialect/python/EnigmaModule.cpp`
- `ref/Enigma-DSL/Enigma-Dialect/lib/CAPI/Dialects.cpp`

Important pattern:

- Ops are defined once in ODS/TableGen under `include/enigma/.../*.td`.
- C++ classes, builders, parsers, printers, verifiers, and enum helpers are generated with `mlir_tablegen`.
- Python dialect bindings are generated from the same ODS file using `declare_mlir_dialect_python_bindings`.
- The Python-side `python/mlir_enigma/dialects/EnigmaOps.td` only includes the real ODS file. It is not a second definition.
- Custom native functions are exposed separately through C API plus nanobind, such as dialect registration, `translate_to_msl`, and `run_standard_pipeline`.

This answers the duplication question: for MLIR ops, yes, use the same `.td` definitions for C++ and Python bindings.

### TileLang

TileLang keeps most of its public language in Python, but its current core path is TVM/TIR-based rather than MLIR-based.

Relevant files:

- `../tilelang/tilelang/language/`
- `../tilelang/tilelang/layout/layout.py`
- `../tilelang/tilelang/engine/lower.py`
- `../tilelang/tilelang/backend/pass_pipeline/pipeline.py`
- `../tilelang/src/transform/`

Important pattern:

- The Python language layer is broad and ergonomic.
- Lowering orchestration is Python-visible, but serious transforms are native/FFI-backed.
- Layout is exposed as a Python object but backed by native functionality.

For TileFlow, mimic the language compatibility, not the TVM IR shape.

### tilelang-mlir-ascend

The Ascend fork has a small MLIR pass library under `tilelangir`, but much of the broader compiler remains TVM/TIR-oriented.

Relevant files:

- `../tilelang-mlir-ascend/tilelangir/include/tilelangir/Transforms/Passes.td`
- `../tilelang-mlir-ascend/tilelangir/include/tilelangir/Transforms/CMakeLists.txt`
- `../tilelang-mlir-ascend/tilelangir/lib/Transforms/CMakeLists.txt`
- `../tilelang-mlir-ascend/tilelangir/python/TileLangIRPasses.cpp`
- `../tilelang-mlir-ascend/tilelangir/tools/tilelangir-opt/tilelangir-opt.cpp`
- `../tilelang-mlir-ascend/src/transform/`

Important pattern:

- Passes are declared with TableGen in `Passes.td`.
- C++ pass implementations are registered into a native optimizer tool.
- Python calls into native pass pipelines through a pybind11 `PassPipeline` wrapper that parses MLIR text, runs textual pass pipelines, and returns MLIR text.

For TileFlow, this is a good model for early pass delegation even before the Python frontend builds MLIR objects directly.

## Single Source of Truth for Operations

Use MLIR ODS/TableGen as the single source of truth for TileFlow operations.

Recommended source layout:

```text
mlir/
  include/tileflow/Dialect/TileFlow/IR/
    TileFlowDialect.td
    TileFlowOps.td
    TileFlowTypes.td
    TileFlowAttrs.td
  lib/Dialect/TileFlow/IR/
    TileFlowDialect.cpp
  include/tileflow/Transforms/
    Passes.td
  lib/Transforms/
    LayoutInference.cpp
    PipelinePlanning.cpp
    LowerToSCF.cpp
  python/
    tileflow_mlir/dialects/TileFlowOps.td  # includes real ODS file
    tileflow_mlir/dialects/tileflow.py
    TileFlowModule.cpp
  tools/tileflow-opt/
```

Generated from the same ODS files:

- C++ op classes and dialect registration.
- Python op builder wrappers.
- Enum attributes.
- Pass declarations from `Passes.td`.

Still likely handwritten:

- Python TileLang-compatible `T` namespace.
- Python AST parser from TileLang syntax to MLIR construction calls.
- High-level compile API.
- C++ pass implementations.
- Runtime bindings.

## Recommended Bootstrap Plan

### Step 1: Replace Python Passes with Native Pass Pipeline Shell

Build a native `tileflow-opt` that registers:

- upstream MLIR dialects
- `tileflow` dialect
- `canonicalize`
- `cse`
- one no-op TileFlow pass

Expose a Python wrapper like Ascend's `PassPipeline.run(mlir_text)`.

At this stage, Python may still emit textual MLIR.

### Step 2: Move Operation Definitions to ODS

Define minimal ops:

- `tileflow.kernel`
- `tileflow.return`
- `tileflow.program_id`
- `tileflow.tensor_decl` or replace with typed function args
- `tileflow.alloc`
- `tileflow.copy`
- `tileflow.gemm`
- `tileflow.parallel`
- `tileflow.pipeline`

Prefer using existing MLIR ops whenever possible:

- arithmetic: `arith`
- loops: `scf` or `affine`
- functions: `func`
- memrefs: `memref`

Do not define custom arithmetic or loop ops unless TileFlow needs different semantics.

### Step 3: Keep Python Frontend Thin

Python frontend should emit MLIR and immediately call native verification/pass pipelines. Native execution is mandatory; Python should not silently optimize or accept frontend IR on its own.

Suggested flow:

```text
TileLang-style Python source
  -> Python AST parser
  -> MLIR module using tileflow + upstream dialects
  -> native tileflow-opt pipeline
  -> optimized/lowered MLIR
  -> backend translation/runtime later
```

### Step 4: Move Analyses Native

Move these out of Python:

- layout inference
- pipeline planning
- memory-space verification
- async copy wait/use verification
- lowering from TileFlow ops to `scf`, `memref`, `gpu`, etc.

Python can keep only temporary debug visualizations of native results.

## Operation Definition Duplication Answer

It is possible, and recommended, to avoid defining MLIR operations twice.

Use this split:

- Define operations once in `.td`.
- Generate C++ op classes from `.td`.
- Generate Python MLIR bindings from the same `.td`.
- Write Python TileLang syntax wrappers separately, but keep them as frontend sugar, not as semantic operation definitions.

The unavoidable distinction:

- `T.copy`, `T.gemm`, `T.Kernel`, etc. are user-facing syntax functions/classes.
- `tileflow.copy`, `tileflow.gemm`, `tileflow.kernel`, etc. are MLIR operations.

Those two layers can have matching names, but they should not both encode full semantics. The `.td` MLIR operation definition should be the semantic source of truth.
