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
    declared_tensors = {
        op.attrs["name"]
        for op in ir.ops
        if op.kind == "tensor_decl" and "name" in op.attrs
    }
    indexed_tensors = set(declared_tensors)
    for op in ir.ops:
        if op.kind == "store" and "target" in op.attrs:
            indexed_tensors.add(str(op.attrs["target"]).split("[", 1)[0])
        if op.kind in {"copy", "async_copy"}:
            for key in ("arg0", "arg1"):
                if key in op.attrs:
                    indexed_tensors.add(str(op.attrs[key]).split("[", 1)[0])
    for arg in ir.args:
        if arg.name not in indexed_tensors:
            continue
        rank = arg.rank or 1
        constraints[arg.name] = LayoutConstraint(
            tensor=arg.name,
            layout=Layout.contiguous(rank),
            reason="default contiguous layout inferred from indexed tensor argument",
        )
    for op in ir.ops:
        if op.kind == "empty" and "name" in op.attrs:
            name = str(op.attrs["name"])
            constraints[name] = LayoutConstraint(
                tensor=name,
                layout=Layout.contiguous(_rank_from_shape(op.attrs.get("arg0")) or 1),
                reason="default contiguous layout inferred from output allocation",
            )
    return constraints


def _rank_from_shape(shape: object) -> int | None:
    if shape is None:
        return None
    text = str(shape)
    if (text.startswith("(") and text.endswith(")")) or (
        text.startswith("[") and text.endswith("]")
    ):
        body = text[1:-1].strip()
        if not body:
            return 0
        return len([part for part in body.split(",") if part.strip()])
    return 1
