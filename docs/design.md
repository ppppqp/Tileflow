# TileFlow Design

TileFlow is a TileLang-compatible Python DSL frontend backed by MLIR. The
source language should stay close to TileLang, but the compiler internals should
not follow TVM/TIR. The implementation should follow the shape used by
`ref/Enigma-DSL`: trace Python syntax into a small Python IR, build a real MLIR
module with MLIR Python bindings, then hand that module to native MLIR/C++ for
verification, optimization, lowering, and target code emission.

## Compiler Pipeline

The intended pipeline is:

```text
TileLang-style Python source
  -> AST rewrite in tileflow.compiler.ast
  -> builder execution / tracing
  -> TileFlow Python IR
  -> MLIR module built with MLIR Python bindings
  -> native MLIR pass pipeline
  -> GPU/NVVM/LLVM lowering
  -> target artifact and runtime wrapper
```

Textual MLIR is a debug/export format, not the primary compiler interface. The
main path should construct an `mlir.ir.Module` directly from the Python IR and
pass `module.operation` to native helpers, matching Enigma's pattern.

## Python Frontend

Python owns source-language compatibility and diagnostics:

- `@tileflow.jit` and `import tileflow.language as T`.
- TileLang-style declarations such as `A: T.Tensor((M, N), T.float32)`.
- AST rewriting for Python constructs that cannot be captured by normal
  operator overloading: assignment, `for`, `if`, `while`, `break`, `continue`,
  `return`, and context managers.
- Builder tracing for expressions, tensor indexing, loads, stores, allocation,
  kernel launch structure, and TileFlow-specific calls.
- Compile-time parameter specialization and source spans for diagnostics.

The AST transformer should rewrite native Python syntax into builder calls. The
builder should record structured SSA-style Python IR with blocks, regions, and
terminators. Python variables are frontend bindings; real IR values are SSA
values.

## Python IR

The Python IR is a frontend trace format, not an optimization IR. It should be
close enough to MLIR that lowering is mechanical:

```text
KernelIR
  params: list[KernelParam]
  body: Region

Region
  blocks: list[Block]

Block
  args: list[Value]
  ops: list[Operation]
  terminator: Operation | None

Operation
  name
  operands
  results
  attrs
  regions
  span

Value
  id
  type
  name_hint
  owner
  span
```

The Python IR should not preserve old compatibility aliases such as
`op.kind`, `op.result`, `kernel.ops`, or stringly typed dtypes. Use one
canonical representation and convert TileLang dtype syntax into real type
objects early.

## MLIR Boundary

TileFlow should use MLIR Python bindings directly, not a custom FFI bridge that
passes Python IR dataclasses into C++.

The boundary should be:

```text
Python IR -> mlir.ir.Module -> native C++ MLIR helpers
```

This follows Enigma:

- Python traced IR is walked in Python.
- MLIR operations are created through MLIR Python bindings.
- Native extension functions receive `MlirOperation` / `module.operation`.
- Native code runs pass pipelines or target translation.
- `str(module)` is used for inspection, tests, and debug dumps.

Avoid this boundary:

```text
Python IR dataclasses -> pybind/nanobind importer -> C++ reconstructs MLIR
```

That would duplicate the IR schema in Python and C++ and make every op, attr,
type, region, terminator, and source span a custom serialization problem.

## ODS And Native Dialect

MLIR ODS/TableGen should be the single source of truth for TileFlow dialect
operations. Define the minimal dialect now, but keep it narrow.

Use upstream MLIR dialects wherever semantics are already standard:

- `arith` for constants, arithmetic, comparisons, and selects.
- `math` for standard math intrinsics.
- `memref` for ordinary loads, stores, subviews, and allocations when possible.
- `scf` for structured control flow when TileFlow does not need custom loop
  semantics.
- `func` for function/module structure.
- `gpu`, `nvgpu`, `nvvm`, and `llvm` for target lowering.

Define TileFlow ops only where TileFlow has domain-specific semantics:

- `tileflow.kernel` for TileFlow kernel boundaries and launch metadata.
- `tileflow.program_id` / `tileflow.thread_id` if launch indices need a stable
  frontend abstraction before GPU lowering.
- `tileflow.parallel` if TileLang parallel-loop semantics carry layout or
  pipeline constraints that are not plain `scf`.
- `tileflow.pipeline` for pipelined loop metadata.
- `tileflow.copy` / `tileflow.async_copy` for copy semantics that later become
  target-specific async operations.
- `tileflow.gemm` for high-level tensor-core or warpgroup operations.
- TileFlow attrs/types for layout, memory space, pipeline stage/order, and
  other compiler-facing annotations.

Avoid begin/end marker ops. Control flow and kernel structure should use
regions.

## Native MLIR Responsibilities

Native MLIR/C++ owns compiler semantics after MLIR construction:

- Dialect op definitions, verifiers, traits, interfaces, parsers, and printers.
- Canonicalization and CSE.
- Layout verification and layout inference.
- Pipeline planning and async-copy legality.
- Memory-space verification and legalization.
- Lowering from TileFlow ops to upstream MLIR dialects.
- Lowering to GPU/NVVM/LLVM and CUDA-oriented emission.
- Tools such as `tileflow-opt` for pass testing and pipeline debugging.

Python may expose convenience wrappers, but it should not become the long-term
home for canonicalization, layout inference, pipeline rewrites, or backend
lowering.

## Lowering Plan

The first vertical slice should be vector add:

```text
Python kernel
  -> Python IR with kernel, parallel loop, load, add, store
  -> MLIR module using tileflow + arith + memref + scf/gpu
  -> native verifier
  -> canonicalize + cse
  -> lower to gpu/nvvm
```

Initial target sequence:

1. Define minimal TileFlow ODS ops with Python bindings.
2. Build a Python MLIR emitter that constructs `mlir.ir.Module` directly.
3. Add native extension helpers for dialect registration and pass execution.
4. Keep textual MLIR output as `str(module)` for debugging.
5. Add `tileflow-opt` tests for the same module shape.
6. Implement one lowering path for vector add.
7. Expand to shared memory, pipelined loops, async copy, and GEMM.

## Enigma Reference

The relevant Enigma pattern is:

```python
module = _build_module(builder)
en.run_standard_pipeline(module.operation)
msl = en.translate_to_msl(module.operation)
```

TileFlow should mirror that structure, replacing Enigma's Metal translator with
TileFlow pass pipelines and CUDA/NVVM-oriented lowering.
