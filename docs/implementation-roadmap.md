# TileFlow Implementation Roadmap

This roadmap breaks TileFlow into small implementation phases. Each step is sized for roughly 2-3 focused days and is meant to produce code, tests, and notes that can later become blog posts, course chapters, or OSS documentation.

The core goal is learning by building:

- Python DSL design and tracing
- compiler IR design
- layout algebra and inference
- pass pipelines
- MLIR dialects, TableGen, conversion, and lowering
- C++/Python FFI
- packaging, runtime dispatch, and documentation

## Reference Map

Use the local checkouts as study references, but keep TileFlow smaller and more explicit.

### Enigma DSL

Study these when building the Python frontend and MLIR boundary:

- `ref/Enigma-DSL/enigma/_tracing.py`: trace-time SSA values, tensor proxies, and operation recording.
- `ref/Enigma-DSL/enigma/compiler/kernel.py`: `@kernel`, `@jit`, annotation handling, and trace orchestration.
- `ref/Enigma-DSL/enigma/compiler/mlir_emitter.py`: Python-to-MLIR emission boundary.
- `ref/Enigma-DSL/Enigma-Dialect/README.md`: native MLIR dialect project shape.
- `ref/Enigma-DSL/Enigma-Dialect/CMakeLists.txt`: MLIR/CMake setup.
- `ref/Enigma-DSL/Enigma-Dialect/test/lit.cfg.py`: lit/FileCheck test setup.

Enigma is useful because it shows a compact Python DSL plus native MLIR dialect split. Its target is Metal-specific; TileFlow should keep the first IR target-neutral.

### TileLang

Study these when building language, layout, lowering, and pipeline structure:

- `../tilelang/tilelang/language/`: user-facing DSL organization.
- `../tilelang/tilelang/layout/layout.py`: layout as a first-class object with FFI-backed behavior.
- `../tilelang/tilelang/engine/lower.py`: target resolution, lowering orchestration, and runtime-facing artifact creation.
- `../tilelang/tilelang/backend/pass_pipeline/pipeline.py`: backend pass pipeline registry.
- `../tilelang/tilelang/jit/`: JIT packaging, caching, and execution boundary.
- `../tilelang/tilelang/carver/`: schedule/search concepts for future autotuning work.

TileLang is useful because it shows how a serious kernel DSL separates language, layout, backend, and JIT layers. It is much larger than TileFlow should be at first.

## Roadmap Shape

Each step should finish with:

- Code in the relevant package.
- At least one test or example.
- A short note in `docs/notes/` capturing what was learned.
- One open question for the next step.

Suggested note format:

```text
# YYYY-MM-DD - Topic

What I built:
What I learned:
What surprised me:
Reference files:
Next question:
```

## Phase 0: Project Foundation

### Step 0.1: Repository Hygiene and Developer Loop

Scope: 2-3 days.

Build:

- Finish basic package metadata in `pyproject.toml`.
- Add `Makefile` or `justfile` commands for format, lint, tests, and examples.
- Add CI later, but design commands so they can run locally first.
- Add `docs/notes/` and a first learning note.

Tests:

- `python3 -m compileall -q src examples tests`
- `python3 -m pytest -q` once pytest is installed.

Learning focus:

- Python package structure.
- Editable installs.
- Keeping compiler experiments reproducible.

Writing output:

- Blog seed: "Setting up a tiny compiler project so experiments stay honest."

### Step 0.2: Design the First Vertical Slice

Scope: 2-3 days.

Build:

- Document the expected end-to-end flow for vector add.
- Freeze a tiny first IR contract: args, values, ops, attributes.
- Decide what is target-neutral and what is backend-specific.
- Add `docs/ir.md` with examples of traced IR and emitted MLIR-like text.

Tests:

- Snapshot test for vector add emitted text.
- Test that unsupported annotations fail with useful diagnostics.

Learning focus:

- Compiler contracts.
- Why early IR stability matters.

Writing output:

- Blog seed: "The smallest useful IR for a Python kernel DSL."

## Phase 1: Python DSL and Tracing

### Step 1.1: Kernel Decorator and Type Annotations

Scope: 2-3 days.

Build:

- Extend `tf.TensorType` with shape, dtype, and optional memory space.
- Add scalar parameter support: `tf.ScalarType("i32")`.
- Add clear error messages for missing or unsupported annotations.
- Keep tracing independent of NumPy, Torch, or device runtimes.

Reference:

- Enigma: `enigma/compiler/kernel.py`
- TileLang: `tilelang/language/kernel.py`

Tests:

- Tensor arg tracing.
- Scalar arg tracing.
- Bad annotation diagnostics.

Learning focus:

- Python introspection.
- Trace-time proxy objects.
- DSL API ergonomics.

Writing output:

- Blog seed: "Decorators are the easy part: what a Python DSL actually captures."

### Step 1.2: Expressions and Primitive Ops

Scope: 2-3 days.

Build:

- Add arithmetic: add, sub, mul, div, mod, neg.
- Add comparisons and boolean operations.
- Add dtype propagation rules.
- Add constants with stable type inference.

Reference:

- Enigma: `enigma/_tracing.py`

Tests:

- Arithmetic op trace order.
- Constant deduping or non-deduping policy.
- Type mismatch diagnostics.

Learning focus:

- Operator overloading limits.
- SSA value modeling.
- Type checking before lowering.

Writing output:

- Blog seed: "How Python operator overloading becomes compiler IR."

### Step 1.3: Tensor Indexing and Memory Ops

Scope: 2-3 days.

Build:

- Support one-dimensional and multi-dimensional indexing.
- Represent load/store index tuples explicitly.
- Add memory spaces: global, shared, local.
- Introduce bounds as metadata, not runtime checks.

Reference:

- TileLang: `tilelang/language/allocate.py`
- TileLang: `tilelang/layout/layout.py`

Tests:

- `a[i]`, `a[i, j]`, and `c[i, j] = value`.
- Rank mismatch diagnostics.
- Memory-space metadata in IR.

Learning focus:

- Tensor semantics.
- Logical indexing versus physical layout.

Writing output:

- Blog seed: "A tensor access is not just a pointer offset."

### Step 1.4: Structured Control Flow

Scope: 2-3 days.

Build:

- Add `tf.range` and `tf.static_range`.
- Add `tf.if_then_else` or an AST preprocessing path for Python `if`.
- Decide whether to trace Python control flow or parse AST for richer syntax.
- Keep a small structured IR: loops and conditionals should be nested regions, not flattened comments.

Reference:

- Enigma: `enigma/compiler/preprocessor.py`
- TileLang: `tilelang/language/loop.py`
- TileLang: `tilelang/language/parser/`

Tests:

- Simple counted loop.
- Static unrolled loop.
- Conditional store.

Learning focus:

- Tracing versus AST parsing.
- Why Python control flow is hard for DSLs.

Writing output:

- Blog seed: "Tracing, AST rewriting, and the control-flow cliff."

## Phase 2: Tile and Layout System

### Step 2.1: Layout Algebra Core

Scope: 2-3 days.

Build:

- Replace the simple `Layout` with explicit input shape, output shape, and index map.
- Add constructors: row-major, column-major, blocked, swizzled.
- Add `compose`, `inverse` where possible, `tile`, and `flatten`.
- Keep symbolic dimensions printable and testable.

Reference:

- Enigma: `enigma/core.py`
- TileLang: `tilelang/layout/layout.py`
- TileLang: `tilelang/layout/swizzle.py`

Tests:

- Row-major and column-major offset calculations.
- Blocked layout mapping.
- Composition smoke tests.

Learning focus:

- Layout as a function.
- Physical memory order versus logical tensor shape.

Writing output:

- Blog seed: "Layouts are functions, not metadata."

### Step 2.2: Access Pattern Collection

Scope: 2-3 days.

Build:

- Add an analysis pass that records every tensor access.
- Classify access patterns: contiguous, strided, broadcast, reduction-like.
- Attach source locations if possible.
- Produce human-readable diagnostics.

Reference:

- TileLang: `tilelang/analysis/`
- TileLang: `tilelang/carver/roller/shape_inference/`

Tests:

- Vector add is contiguous.
- Matrix transpose is strided.
- Reduction reads many values and writes one value.

Learning focus:

- Compiler analysis as facts plus confidence.
- Diagnostics as first-class compiler output.

Writing output:

- Blog seed: "What the compiler can infer from indexing alone."

### Step 2.3: First Layout Inference Pass

Scope: 2-3 days.

Build:

- Infer default physical layouts for tensor args from rank and access pattern.
- Represent constraints and conflicts explicitly.
- Add user layout annotations that override inference.
- Emit layout attributes in MLIR-like output.

Tests:

- Inferred contiguous vector layout.
- Inferred row-major matrix layout.
- User annotation wins over default.
- Conflicting constraints produce diagnostics.

Learning focus:

- Constraint solving without overengineering.
- Making inference explainable.

Writing output:

- Blog seed: "A tiny layout inference engine."

### Step 2.4: Shared Memory and Tile Views

Scope: 2-3 days.

Build:

- Add `tf.alloc_shared(shape, dtype, layout=...)`.
- Add tile views over global tensors.
- Model copy from global tile to shared tile.
- Keep actual async behavior for a later phase.

Reference:

- TileLang: `tilelang/language/allocate.py`
- TileLang: `tilelang/language/copy_op.py`

Tests:

- Shared allocation appears in IR.
- Copy op carries source and destination layouts.
- Rank and shape mismatch diagnostics.

Learning focus:

- Memory hierarchy modeling.
- Why tiling belongs before backend lowering.

Writing output:

- Blog seed: "Introducing shared memory without writing a backend yet."

## Phase 3: Pass Infrastructure

### Step 3.1: Pass Manager

Scope: 2-3 days.

Build:

- Replace ad hoc compile orchestration with a `PassManager`.
- Define pass inputs, outputs, preserved analyses, and diagnostics.
- Add pass timing and text dumps.
- Add `TF_DUMP_IR=1` or compile options.

Reference:

- TileLang: `backend/pass_pipeline/pipeline.py`
- TileLang: `engine/lower.py`

Tests:

- Pass order is deterministic.
- Failed pass reports which pass failed.
- Dump output includes pass names.

Learning focus:

- Compiler pipeline architecture.
- Debuggability as infrastructure.

Writing output:

- Blog seed: "A pass manager before the compiler gets complicated."

### Step 3.2: Verification Passes

Scope: 2-3 days.

Build:

- Add IR verifier: values defined before use, valid tensor args, valid attrs.
- Add type verifier.
- Add layout verifier.
- Run verifier before and after major passes.

Tests:

- Manually constructed invalid IR cases.
- User-facing DSL errors remain readable.

Learning focus:

- Invariants.
- Why compilers verify constantly.

Writing output:

- Blog seed: "Compiler verification is where good errors come from."

### Step 3.3: Canonicalization and Simplification

Scope: 2-3 days.

Build:

- Fold simple constants.
- Remove dead operations.
- Normalize index expressions.
- Add deterministic IR printing.

Tests:

- `i + 0` simplifies.
- Unused load is removed if side-effect free.
- Stores are never removed by accident.

Learning focus:

- Side effects.
- Canonical forms.
- Why optimization starts small.

Writing output:

- Blog seed: "The first optimization pass in a tiny compiler."

## Phase 4: Pipelining Model

### Step 4.1: Pipeline IR

Scope: 2-3 days.

Build:

- Replace coarse comments with explicit pipeline stage objects in IR.
- Add operations: `pipeline.begin`, `pipeline.copy`, `pipeline.wait`, `pipeline.compute`, `pipeline.commit`.
- Model dependencies between stages.
- Keep this target-neutral.

Tests:

- Load/compute/store vector add gets three stages.
- Tiled copy/compute kernel gets copy and compute stages.
- Dependency graph is acyclic.

Learning focus:

- Scheduling versus lowering.
- Data dependency modeling.

Writing output:

- Blog seed: "Pipelining as an IR problem."

### Step 4.2: Double Buffering

Scope: 2-3 days.

Build:

- Add `num_stages` and buffer slot analysis.
- Rewrite tiled loop bodies into prologue, steady-state, epilogue structure.
- Emit readable MLIR-like text showing pipeline slots.

Reference:

- TileLang: CUDA pipeline utilities under `tilelang/cuda/pipeline.py`

Tests:

- Two-stage pipeline assigns alternating slots.
- Prologue and epilogue are emitted.
- Invalid stage count errors clearly.

Learning focus:

- Software pipelining.
- Latency hiding.

Writing output:

- Blog seed: "Double buffering from first principles."

### Step 4.3: Async Copy Semantics

Scope: 2-3 days.

Build:

- Add async copy operation to DSL.
- Track copy groups and waits.
- Keep backend-specific lowering abstract.
- Add target capability flags so unsupported async copy falls back or errors.

Reference:

- TileLang: `tilelang/language/copy_op.py`
- Enigma: `tests/portable/test_async_copy.py`

Tests:

- Async copy produces group metadata.
- Missing wait before use is diagnosed.
- Unsupported target capability errors.

Learning focus:

- Memory ordering.
- Semantic checks before backend lowering.

Writing output:

- Blog seed: "Async copy is a contract, not just an instruction."

## Phase 5: Textual MLIR and Dialect Design

### Step 5.1: Realistic MLIR Text Emission

Scope: 2-3 days.

Build:

- Emit syntactically closer MLIR text with types, attributes, and regions.
- Separate generic MLIR emission from TileFlow dialect op names.
- Add FileCheck-style expected output files even before native tools exist.

Reference:

- Enigma: `enigma/compiler/mlir_emitter.py`
- Enigma-Dialect: `test/`

Tests:

- Golden files under `tests/mlir/`.
- Stable output after canonicalization.

Learning focus:

- MLIR syntax.
- Regions, attributes, types, and assembly forms.

Writing output:

- Blog seed: "Learning MLIR by emitting text first."

### Step 5.2: Dialect Operation Spec

Scope: 2-3 days.

Build:

- Add `docs/dialect.md`.
- Specify initial ops: kernel, program_id, load, store, alloc, layout_map, pipeline ops.
- Specify traits, side effects, attributes, and result types.
- Write example MLIR manually.

Reference:

- Enigma-Dialect: op list in `README.md`
- MLIR ODS/TableGen style from Enigma dialect files once present in local checkout.

Tests:

- Not code-heavy; review examples for consistency.

Learning focus:

- Dialect design.
- Separating source DSL concepts from IR concepts.

Writing output:

- Blog seed: "Designing an MLIR dialect before writing C++."

### Step 5.3: Native Dialect Skeleton

Scope: 2-3 days.

Build:

- Create `mlir/` or `dialect/` native subproject.
- Add CMake, TableGen files, dialect registration, and one op.
- Build `tileflow-opt`.
- Add lit test for parse/print roundtrip.

Reference:

- Enigma-Dialect root structure.
- `Enigma-Dialect/CMakeLists.txt`
- `Enigma-Dialect/test/lit.cfg.py`

Tests:

- `ninja check-tileflow`
- Parse/print test for `tileflow.kernel`.

Learning focus:

- MLIR build system.
- ODS/TableGen.
- lit and FileCheck.

Writing output:

- Blog seed: "My first MLIR dialect: CMake, TableGen, and one op."

### Step 5.4: Add Core Dialect Ops

Scope: 2-3 days.

Build:

- Add program ID, load, store, return, and simple arithmetic ops or decide to reuse `arith` for arithmetic.
- Implement verifiers where useful.
- Add parse/print tests.

Tests:

- Valid examples parse.
- Invalid examples fail with expected diagnostics.

Learning focus:

- Operation traits.
- Custom verification.
- Reusing standard MLIR dialects.

Writing output:

- Blog seed: "When to create a custom op and when to reuse MLIR."

## Phase 6: Python to Native MLIR

### Step 6.1: MLIR Python Bindings Strategy

Scope: 2-3 days.

Build:

- Decide whether to start with subprocess text roundtrip or Python bindings.
- Document local LLVM/MLIR setup.
- Add a script to run `tileflow-opt` on emitted MLIR.
- Keep the Python package usable without native dependencies.

Reference:

- Enigma-Dialect Python binding notes.
- Enigma build scripts.

Tests:

- Python emits MLIR text.
- Native tool parses it when available.
- Tests skip cleanly when native tools are absent.

Learning focus:

- Optional native dependencies.
- Developer experience around MLIR.

Writing output:

- Blog seed: "Bridging Python and MLIR without making imports painful."

### Step 6.2: Python FFI Smoke Binding

Scope: 2-3 days.

Build:

- Expose one native function to Python: register dialect or parse module.
- Package it as an optional extension.
- Add import tests.

Reference:

- Enigma-Dialect: `python/EnigmaModule.cpp`
- TileLang: FFI-backed layout in `tilelang/layout/layout.py`

Tests:

- Import binding.
- Call native function.
- Skip if extension is not built.

Learning focus:

- C++/Python extension boundaries.
- ABI and packaging pain points.

Writing output:

- Blog seed: "The first FFI bridge in a compiler project."

### Step 6.3: Replace Text Emitter Internals

Scope: 2-3 days.

Build:

- Build MLIR modules through Python bindings or native helper APIs.
- Keep textual output as a debugging/export format.
- Add roundtrip tests against `tileflow-opt`.

Tests:

- Python IR to native MLIR module.
- Native parse/print roundtrip.
- Golden output remains stable enough for tests.

Learning focus:

- MLIR object model.
- Contexts, locations, insertion points, and ownership.

Writing output:

- Blog seed: "From strings to real MLIR modules."

## Phase 7: Lowering and Backend Experiments

### Step 7.1: Lower to Standard MLIR Dialects

Scope: 2-3 days.

Build:

- Lower TileFlow ops to `func`, `scf`, `arith`, `memref`, and maybe `gpu`.
- Add conversion patterns.
- Add pass pipeline in native tools.

Tests:

- `tileflow-opt --convert-tileflow-to-standard`.
- FileCheck output uses standard dialects.

Learning focus:

- MLIR conversion patterns.
- Type conversion.
- Legality.

Writing output:

- Blog seed: "Lowering a custom dialect into standard MLIR."

### Step 7.2: CPU Interpreter or LLVM Path

Scope: 2-3 days.

Build:

- Pick the simplest executable path: interpreter, C emitter, or LLVM lowering.
- Run vector add on CPU first.
- Return results to Python.

Tests:

- Vector add correctness.
- Scalar args correctness.

Learning focus:

- Execution before GPU.
- Runtime ABI.

Writing output:

- Blog seed: "Making the compiler run something, even on CPU."

### Step 7.3: GPU Dialect Lowering Sketch

Scope: 2-3 days.

Build:

- Map kernel launch concepts to MLIR `gpu` dialect.
- Represent block/thread IDs.
- Lower memory spaces to address spaces.
- Do not chase high performance yet.

Tests:

- FileCheck for `gpu.module`, `gpu.func`, and ID ops.

Learning focus:

- MLIR GPU dialect.
- Target-independent GPU representation.

Writing output:

- Blog seed: "A small Python DSL meets MLIR's GPU dialect."

### Step 7.4: Target-Specific Backend Spike

Scope: 2-3 days.

Build:

- Choose one backend spike: CUDA/NVVM, ROCm/ROCDL, Metal-style emitter, or CPU.
- Define the minimum vector add lowering.
- Document every missing piece.

Reference:

- Enigma for Metal-style custom emitter.
- TileLang for target resolution and runtime compilation.

Tests:

- Native tool emits target code or reaches known unsupported point.

Learning focus:

- Backend reality check.
- Target-specific ABI and toolchains.

Writing output:

- Blog seed: "The first backend spike: what broke and what became clear."

## Phase 8: Runtime, JIT, and Packaging

### Step 8.1: Compiled Artifact Model

Scope: 2-3 days.

Build:

- Define `CompiledKernel` fields for IR, MLIR, target code, binary path, params, and diagnostics.
- Add cache keys based on source, options, and target.
- Add export methods.

Reference:

- TileLang: `engine/param.py`
- TileLang: `cache/`

Tests:

- Cache key changes when source/options change.
- Export writes expected files.

Learning focus:

- Compiler products.
- Reproducibility and caching.

Writing output:

- Blog seed: "What does a compiler return?"

### Step 8.2: Runtime Invocation Boundary

Scope: 2-3 days.

Build:

- Define launch config and argument binding API.
- Add CPU or mock runtime first.
- Add clear errors for mismatched arguments.

Reference:

- TileLang: `jit/adapter/`
- Enigma: `runtime_dispatch/`

Tests:

- Bind args by position.
- Bind args by name.
- Argument count and dtype errors.

Learning focus:

- Runtime ABI.
- Where compiler ends and runtime begins.

Writing output:

- Blog seed: "The runtime boundary of a JIT compiler."

### Step 8.3: Wheel and Optional Native Build

Scope: 2-3 days.

Build:

- Keep pure Python install simple.
- Add optional native extension build path.
- Document local MLIR requirements.
- Decide whether native artifacts live in same package or separate package.

Reference:

- Enigma split between Python DSL and dialect wheel.
- TileLang scikit-build setup.

Tests:

- Pure Python import works without MLIR.
- Native tests run only when extension exists.

Learning focus:

- Python packaging for compilers.
- Optional native dependencies.

Writing output:

- Blog seed: "Packaging a compiler that has both Python and C++."

## Phase 9: Examples and Case Studies

### Step 9.1: Vector Add Deep Dive

Scope: 2-3 days.

Build:

- Make vector add the canonical tutorial.
- Show Python DSL, traced IR, inferred layout, pipeline, MLIR, and lowered output.
- Add comments only in docs, not noisy runtime output.

Tests:

- Example stays runnable.
- Expected output remains stable.

Learning focus:

- End-to-end explanation.

Writing output:

- Blog/course chapter: "Vector add through every layer of TileFlow."

### Step 9.2: Matrix Transpose

Scope: 2-3 days.

Build:

- Add transpose example.
- Use it to demonstrate non-contiguous access and layout inference.
- Add shared memory tile variant if Phase 2.4 is complete.

Tests:

- Inference identifies strided access.
- Emitted layout metadata differs from vector add.

Learning focus:

- Access patterns and memory coalescing.

Writing output:

- Blog/course chapter: "Transpose: the first layout-sensitive kernel."

### Step 9.3: Tiled Matmul Skeleton

Scope: 2-3 days.

Build:

- Add a minimal tiled matmul DSL example.
- Represent shared memory tiles and pipeline stages.
- Do not require peak performance.

Reference:

- TileLang matmul templates in `tilelang/carver/template/matmul.py`.

Tests:

- Trace and pipeline shape are correct.
- Diagnostics catch incompatible tile shapes.

Learning focus:

- Tiling.
- Accumulators.
- Shared memory staging.

Writing output:

- Blog/course chapter: "Matmul as a compiler design test."

### Step 9.4: Reduction Kernel

Scope: 2-3 days.

Build:

- Add row-sum or RMSNorm-style reduction.
- Add reduction op or pattern.
- Model local/shared memory accumulation.

Reference:

- Enigma RMSNorm examples.
- TileLang reduce ops.

Tests:

- Reduction trace contains explicit reduction structure.
- Layout inference recognizes read-many/write-one pattern.

Learning focus:

- Associativity.
- Parallel reductions.

Writing output:

- Blog/course chapter: "Reductions force the DSL to grow up."

## Phase 10: OSS or Course Readiness

### Step 10.1: Documentation Architecture

Scope: 2-3 days.

Build:

- Organize docs into tutorial, design notes, reference, and internals.
- Move learning notes into polished articles.
- Add diagrams for pipeline and layering.

Tests:

- Documentation examples run.
- Links are valid.

Learning focus:

- Teaching compiler systems.
- Turning experiments into durable material.

Writing output:

- Course outline and first module.

### Step 10.2: Contributor and Project Standards

Scope: 2-3 days.

Build:

- Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, and license review.
- Add clear "project status" labels.
- Document what is stable and what is experimental.

Tests:

- Fresh clone setup instructions work.

Learning focus:

- OSS expectations.
- Reducing contributor confusion.

Writing output:

- Blog seed: "Turning a learning compiler into an OSS project."

### Step 10.3: Course Project Packaging

Scope: 2-3 days.

Build:

- Convert phases into lessons.
- Add exercises at the end of each lesson.
- Mark stretch goals.
- Add expected outputs but avoid giving away every implementation detail.

Tests:

- A learner can complete Phase 1 without native MLIR.
- A learner can complete MLIR phases with documented prerequisites.

Learning focus:

- Curriculum design.
- Progressive disclosure.

Writing output:

- Public course landing page draft.

## Suggested First 30 Days

Week 1:

- Step 0.1: Developer loop.
- Step 0.2: First vertical slice design.
- Step 1.1: Kernel annotations.

Week 2:

- Step 1.2: Expressions.
- Step 1.3: Tensor indexing.
- Step 1.4: Structured control-flow design or first implementation.

Week 3:

- Step 2.1: Layout algebra core.
- Step 2.2: Access pattern collection.
- Step 2.3: First layout inference.

Week 4:

- Step 3.1: Pass manager.
- Step 3.2: Verification passes.
- Step 5.1: Realistic MLIR text emission.

This order keeps the project useful before native MLIR is ready. The first serious milestone is not "runs on GPU"; it is "a Python kernel traces into a verified IR, gets layout constraints, gets pipeline metadata, and emits MLIR text that explains itself."

## Decision Points

These decisions should be made deliberately and documented when reached:

- Tracing only versus AST parsing for Python syntax.
- Textual MLIR first versus MLIR Python bindings first.
- Single package versus split Python/native packages.
- Custom backend emitter versus lowering through upstream MLIR dialects.
- CPU-first runtime versus GPU-first runtime.
- Layout inference as simple constraints versus solver-backed inference.

## Blog and Course Arc

A natural content sequence:

1. Why build a tiny DSL compiler?
2. Python tracing and SSA IR.
3. Tensor indexing and layouts.
4. Layout inference.
5. Pass managers and compiler diagnostics.
6. MLIR text emission.
7. Building the first MLIR dialect.
8. Python/C++ FFI for compilers.
9. Lowering to standard dialects.
10. Runtime and JIT packaging.
11. Case study: transpose.
12. Case study: tiled matmul.

The most valuable insights will come from comparing three representations of the same kernel: the Python DSL, TileFlow IR, and MLIR. Preserve those snapshots as the project evolves.

