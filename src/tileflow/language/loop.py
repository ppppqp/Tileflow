from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tileflow.language.ir import LoopKind, ValueLike


@dataclass(frozen=True)
class ForDim:
    lo: ValueLike
    hi: ValueLike
    step: ValueLike = 1


@dataclass(frozen=True)
class ForSpec:
    dims: tuple[ForDim, ...]
    kind: LoopKind = "serial"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)


def make_range_spec(*args: ValueLike, kind: LoopKind = "serial") -> ForSpec:
    if len(args) == 1:
        lo, hi, step = 0, args[0], 1
    elif len(args) == 2:
        lo, hi, step = args[0], args[1], 1
    elif len(args) == 3:
        lo, hi, step = args
    else:
        raise TypeError(f"range expected 1 to 3 arguments, got {len(args)}")
    return ForSpec(dims=(ForDim(lo=lo, hi=hi, step=step),), kind=kind)


def Parallel(*extents: ValueLike, **attrs: Any) -> ForSpec:
    if not extents:
        raise TypeError("T.Parallel expected at least one extent")
    return ForSpec(
        dims=tuple(ForDim(lo=0, hi=extent, step=1) for extent in extents),
        kind="parallel",
        attrs=dict(attrs),
    )


def Serial(*args: ValueLike, **attrs: Any) -> ForSpec:
    spec = make_range_spec(*args, kind="serial")
    return ForSpec(dims=spec.dims, kind=spec.kind, attrs=dict(attrs))


def Pipelined(
    start: ValueLike,
    stop: ValueLike | None = None,
    step: ValueLike = 1,
    **attrs: Any,
) -> ForSpec:
    if stop is None:
        lo, hi = 0, start
    else:
        lo, hi = start, stop
    return ForSpec(dims=(ForDim(lo=lo, hi=hi, step=step),), kind="pipelined", attrs=dict(attrs))
