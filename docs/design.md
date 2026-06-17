# TileFlow Design Notes

For the implementation plan, see [implementation-roadmap.md](implementation-roadmap.md).

## Frontend

The Python frontend follows the same broad shape as Enigma DSL: a decorator wraps a Python function, tracing replaces tensor arguments with proxy objects, and operator overloads record SSA-style operations.

Unlike Enigma's Metal-specific surface, TileFlow should keep target details out of the first IR. Target lowering should happen after layout inference and pipeline planning.

## Layout Inference

Layouts are represented as constraints. This keeps inference separate from mutation and makes diagnostics easier:

- Every tensor access can contribute a desired layout.
- Transforming operations can propagate or refine a layout.
- Conflicts can be reported with the operations that introduced them.

The bootstrap implementation only assigns contiguous layouts to indexed tensor arguments.

## Pipelining

The first pipeline pass groups loads, compute, and stores into coarse stages. The intended direction is to model async copies, waits, commit groups, and double buffering before MLIR emission.

## MLIR Boundary

The text emitter currently produces MLIR-like `tileflow.*` operations for design iteration. The native milestone is a real MLIR dialect with Python bindings, following the split used by `ref/Enigma-DSL/Enigma-Dialect`.
