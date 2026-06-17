# Step 3.1: Pass Manager

Scope: 2-3 days.

Goal: make the compiler pipeline explicit, ordered, inspectable, and easy to extend.

Build:

- Replace ad hoc compile orchestration with a `PassManager`.
- Define pass inputs, outputs, preserved analyses, and diagnostics.
- Add pass timing and text dumps.
- Add `TF_DUMP_IR=1` or equivalent compile options.

Tests:

- Pass order is deterministic.
- Failed pass reports which pass failed.
- Dump output includes pass names.

References:

- `../tilelang/tilelang/backend/pass_pipeline/pipeline.py`
- `../tilelang/tilelang/engine/lower.py`

Learning focus:

- Compiler pipeline architecture.
- Debuggability as infrastructure.

Writing output:

- Blog seed: "A pass manager before the compiler gets complicated."

