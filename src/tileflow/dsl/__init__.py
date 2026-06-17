"""TileLang-compatible Python DSL frontend.

The public syntax is intentionally TileLang-like:

    import tileflow
    import tileflow.language as T

    @tileflow.jit
    def kernel(A, B, block_M: int):
        M, N = T.const("M, N")
        A: T.Tensor((M, N), T.float16)
        ...

Unlike TileLang, this frontend parses the Python AST into TileFlow's own
MLIR-oriented IR instead of producing TVM TIR.
"""

from .jit import JitFunction, jit

__all__ = ["JitFunction", "jit"]

