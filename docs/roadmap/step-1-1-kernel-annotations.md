# Step 1.1: Kernel Decorator and Type Annotations

Scope: 2-3 days.

Goal: make function signatures carry enough information to trace kernels predictably.

Build:

- Extend `tf.TensorType` with shape, dtype, and optional memory space.
- Add scalar parameter support with `tf.ScalarType("i32")`.
- Add diagnostics for missing annotations and unsupported annotations.
- Keep tracing independent of NumPy, Torch, MLIR, or device runtimes.

Tests:

- Tensor argument tracing.
- Scalar argument tracing.
- Bad annotation diagnostics.

References:

- `ref/Enigma-DSL/enigma/compiler/kernel.py`
- `../tilelang/tilelang/language/kernel.py`

Learning focus:

- Python introspection.
- Trace-time proxy objects.
- DSL API ergonomics.

Writing output:

- Blog seed: "Decorators are the easy part: what a Python DSL actually captures."

