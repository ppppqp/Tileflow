# TileFlow Roadmap Index

TileFlow is a learning compiler project for Python DSLs, layout inference, pass pipelines, MLIR, FFI, and runtime/JIT packaging. The roadmap is intentionally split into small 2-3 day steps so each unit can become a durable implementation note, blog post, or course lesson.

## Reference Map

### Enigma DSL

- `ref/Enigma-DSL/enigma/_tracing.py`: trace-time SSA values, tensor proxies, and operation recording.
- `ref/Enigma-DSL/enigma/compiler/kernel.py`: `@kernel`, `@jit`, annotation handling, and trace orchestration.
- `ref/Enigma-DSL/enigma/compiler/mlir_emitter.py`: Python-to-MLIR emission boundary.
- `ref/Enigma-DSL/Enigma-Dialect/README.md`: native MLIR dialect project shape.
- `ref/Enigma-DSL/Enigma-Dialect/CMakeLists.txt`: MLIR/CMake setup.
- `ref/Enigma-DSL/Enigma-Dialect/test/lit.cfg.py`: lit/FileCheck test setup.

### TileLang

- `../tilelang/tilelang/language/`: user-facing DSL organization.
- `../tilelang/tilelang/layout/layout.py`: layout as a first-class object with FFI-backed behavior.
- `../tilelang/tilelang/engine/lower.py`: target resolution, lowering orchestration, and runtime-facing artifact creation.
- `../tilelang/tilelang/backend/pass_pipeline/pipeline.py`: backend pass pipeline registry.
- `../tilelang/tilelang/jit/`: JIT packaging, caching, and execution boundary.
- `../tilelang/tilelang/carver/`: schedule/search concepts for future autotuning work.

## Phases

- [Phase 0: Project Foundation](phase-0-foundation.md)
- [Phase 1: Python DSL and Tracing](phase-1-python-dsl.md)
- [Phase 2: Tile and Layout System](phase-2-layout-system.md)
- [Phase 3: Pass Infrastructure](phase-3-pass-infrastructure.md)
- [Phase 4: Pipelining Model](phase-4-pipelining.md)
- [Phase 5: Textual MLIR and Dialect Design](phase-5-mlir-dialect.md)
- [Phase 6: Python to Native MLIR](phase-6-python-native-mlir.md)
- [Phase 7: Lowering and Backend Experiments](phase-7-lowering-backends.md)
- [Phase 8: Runtime, JIT, and Packaging](phase-8-runtime-jit-packaging.md)
- [Phase 9: Examples and Case Studies](phase-9-examples.md)
- [Phase 10: OSS or Course Readiness](phase-10-oss-course.md)

## Suggested First 30 Days

Week 1:

- [Step 0.1: Repository Hygiene and Developer Loop](step-0-1-repository-hygiene.md)
- [Step 0.2: Design the First Vertical Slice](step-0-2-first-vertical-slice.md)
- [Step 1.1: Kernel Decorator and Type Annotations](step-1-1-kernel-annotations.md)

Week 2:

- [Step 1.2: Expressions and Primitive Ops](step-1-2-expressions.md)
- [Step 1.3: Tensor Indexing and Memory Ops](step-1-3-tensor-indexing.md)
- [Step 1.4: Structured Control Flow](step-1-4-control-flow.md)

Week 3:

- [Step 2.1: Layout Algebra Core](step-2-1-layout-algebra.md)
- [Step 2.2: Access Pattern Collection](step-2-2-access-patterns.md)
- [Step 2.3: First Layout Inference Pass](step-2-3-layout-inference.md)

Week 4:

- [Step 3.1: Pass Manager](step-3-1-pass-manager.md)
- [Step 3.2: Verification Passes](step-3-2-verification.md)
- [Step 5.1: Realistic MLIR Text Emission](step-5-1-realistic-mlir-text.md)

## Step Note Template

```text
# YYYY-MM-DD - Topic

What I built:
What I learned:
What surprised me:
Reference files:
Next question:
```
