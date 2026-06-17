# TileFlow Implementation Roadmap

This roadmap is split into per-step implementation guides under [`roadmap/`](roadmap/).

Each step is intended to take roughly 2-3 focused days and should produce:

- Code in the relevant package.
- At least one test or example.
- A short learning note under `docs/notes/`.    
- One open question for the next step.

## Start Here

- [Roadmap Index](roadmap/index.md)
- [Phase 0: Project Foundation](roadmap/phase-0-foundation.md)
- [Phase 1: Python DSL and Tracing](roadmap/phase-1-python-dsl.md)
- [Phase 2: Tile and Layout System](roadmap/phase-2-layout-system.md)
- [Phase 3: Pass Infrastructure](roadmap/phase-3-pass-infrastructure.md)
- [Phase 4: Pipelining Model](roadmap/phase-4-pipelining.md)
- [Phase 5: Textual MLIR and Dialect Design](roadmap/phase-5-mlir-dialect.md)
- [Phase 6: Python to Native MLIR](roadmap/phase-6-python-native-mlir.md)
- [Phase 7: Lowering and Backend Experiments](roadmap/phase-7-lowering-backends.md)
- [Phase 8: Runtime, JIT, and Packaging](roadmap/phase-8-runtime-jit-packaging.md)
- [Phase 9: Examples and Case Studies](roadmap/phase-9-examples.md)
- [Phase 10: OSS or Course Readiness](roadmap/phase-10-oss-course.md)

## Reference Projects

Use the local checkouts as study references:

- `ref/Enigma-DSL`: compact Python DSL tracing, MLIR emission boundary, and native dialect split.
- `../tilelang`: larger production-style organization across language, layout, backend, lowering, JIT, and runtime layers.

The first serious milestone is not "runs on GPU"; it is "a Python kernel traces into a verified IR, gets layout constraints, gets pipeline metadata, and emits MLIR text that explains itself."
