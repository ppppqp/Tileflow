"""Layout data structures and bootstrap inference rules."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import KernelIR


@dataclass(frozen=True)
class Layout:
    """A logical shape/stride mapping."""

    shape: tuple[int | str, ...]
    stride: tuple[int | str, ...]
    memory_space: str = "global"

    @classmethod
    def contiguous(cls, rank: int, *, memory_space: str = "global") -> "Layout":
        shape = tuple(f"d{i}" for i in range(rank))
        stride: list[int | str] = []
        for axis in range(rank):
            if axis == rank - 1:
                stride.append(1)
            else:
                stride.append("*".join(f"d{i}" for i in range(axis + 1, rank)))
        return cls(shape=shape, stride=tuple(stride), memory_space=memory_space)


@dataclass(frozen=True)
class LayoutConstraint:
    tensor: str
    layout: Layout
    reason: str


def infer_layouts(ir: KernelIR) -> dict[str, LayoutConstraint]:
    """Infer default layouts from argument ranks and access metadata.

    This first pass deliberately uses conservative contiguous layouts. The
    function returns constraints rather than mutating the IR so later passes can
    attach diagnostics and resolve conflicts.
    """

    constraints: dict[str, LayoutConstraint] = {}
    indexed_tensors = {
        op.attrs["tensor"]
        for op in ir.ops
        if op.kind in {"load", "store"} and "tensor" in op.attrs
    }
    for arg in ir.args:
        if arg.name not in indexed_tensors:
            continue
        rank = arg.rank or 1
        constraints[arg.name] = LayoutConstraint(
            tensor=arg.name,
            layout=Layout.contiguous(rank),
            reason="default contiguous layout inferred from indexed tensor argument",
        )
    return constraints

