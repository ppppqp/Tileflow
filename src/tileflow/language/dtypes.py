import numpy as np


class dtype:
    @property
    def bits(self) -> int:
        raise NotImplementedError

    @property
    def bytes(self) -> int:
        raise NotImplementedError


_PYTHON_DTYPE_TO_STR = {bool: "bool", int: "int32", float: "float32"}
_NUMPY_DTYPE_TO_STR = {
    np.bool_: "bool",
    np.int32: "int32",
    np.float16: "float16",
    np.float32: "float32",
}

_NUMPY_DTYPE_TO_STR.update({np.dtype(k): v for k, v in _NUMPY_DTYPE_TO_STR.items()})


class bool(dtype):
    @property
    def bits(self) -> int:
        return 1

    @property
    def bytes(self) -> int:
        return 1


class int32(dtype):
    @property
    def bits(self) -> int:
        return 32

    @property
    def bytes(self) -> int:
        return 4


class float16(dtype):
    @property
    def bits(self) -> int:
        return 16

    @property
    def bytes(self) -> int:
        return 2


class float32(dtype):
    @property
    def bits(self) -> int:
        return 32

    @property
    def bytes(self) -> int:
        return 4


_all_dtypes = [
    "bool",
    "int32",
    "float16",
    "float32",
]

__all__ = list(_all_dtypes) + [
    "dtype",
]
