# Step 7.4: Target-Specific Backend Spike

Scope: 2-3 days.

Goal: test one concrete backend path and document what breaks.

Build:

- Choose one backend spike: CUDA/NVVM, ROCm/ROCDL, Metal-style emitter, or CPU.
- Define the minimum vector-add lowering.
- Document every missing piece.

Tests:

- Native tool emits target code or reaches a known unsupported point.

References:

- Enigma for Metal-style custom emitter.
- TileLang for target resolution and runtime compilation.

Learning focus:

- Backend ABI.
- Toolchain constraints.
- Scope control.

Writing output:

- Blog seed: "The first backend spike: what broke and what became clear."

