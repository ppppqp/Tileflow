# Step 8.2: Runtime Invocation Boundary

Scope: 2-3 days.

Goal: define the API between compiled kernels and runtime execution.

Build:

- Define launch config and argument binding API.
- Add CPU or mock runtime first.
- Add clear errors for mismatched arguments.

Tests:

- Bind args by position.
- Bind args by name.
- Argument count and dtype errors.

References:

- `../tilelang/tilelang/jit/adapter/`
- `ref/Enigma-DSL/enigma/runtime_dispatch/`

Learning focus:

- Runtime ABI.
- Where compiler ends and runtime begins.

Writing output:

- Blog seed: "The runtime boundary of a JIT compiler."

