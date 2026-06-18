"""TileLang-compatible language namespace.

These objects exist so TileFlow programs can use the same source-level shape
as TileLang programs. The compiler currently consumes them from Python AST
rather than executing most of them at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


float16 = "float16"
float32 = "float32"
float64 = "float64"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
bool = "bool"


@dataclass(frozen=True)
class TensorSpec:
    shape: Any
    dtype: Any


def Tensor(shape: Any, dtype: Any) -> TensorSpec:
    return TensorSpec(shape, dtype)


def const(names: str):
    return tuple(name.strip() for name in names.split(","))


def ceildiv(a: Any, b: Any) -> str:
    return f"ceildiv({a}, {b})"


class Kernel:
    def __init__(self, *grid: Any, threads: Any = None, cluster_dims: Any = None):
        self.grid = grid
        self.threads = threads
        self.cluster_dims = cluster_dims

    def __enter__(self):
        names = [f"b{i}" for i in range(len(self.grid))]
        if len(names) == 1:
            return names[0]
        return tuple(names)

    def __exit__(self, exc_type, exc, tb):
        return False


def Pipelined(extent: Any, *, num_stages: int = 1):
    return range(0)


def Parallel(*extents: Any):
    return range(0)


def Sequential(*extents: Any):
    return range(0)


def alloc_shared(shape: Any, dtype: Any, scope: str = "shared"):
    return {"kind": "alloc_shared", "shape": shape, "dtype": dtype, "scope": scope}


def alloc_local(shape: Any, dtype: Any, scope: str = "local"):
    return {"kind": "alloc_local", "shape": shape, "dtype": dtype, "scope": scope}


def alloc_fragment(shape: Any, dtype: Any, scope: str = "local.fragment"):
    return {"kind": "alloc_fragment", "shape": shape, "dtype": dtype, "scope": scope}


def empty(shape: Any, dtype: Any):
    return {"kind": "empty", "shape": shape, "dtype": dtype, "scope": "global"}


def copy(src: Any, dst: Any, **kwargs: Any) -> None:
    return None


def async_copy(src: Any, dst: Any, **kwargs: Any) -> None:
    return None


def gemm(a: Any, b: Any, c: Any, **kwargs: Any) -> None:
    return None


def clear(buffer: Any) -> None:
    return None


def max(a: Any, b: Any):
    return a if a >= b else b
