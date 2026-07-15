import builtins

import numpy as np

from tileflow.language.ir import BoolType, FloatType, IntType, Type


dtype = Type

bool = BoolType()
int32 = IntType(32)
float16 = FloatType(16)
float32 = FloatType(32)

_PYTHON_DTYPE_TO_STR = {builtins.bool: "bool", builtins.int: "int32", builtins.float: "float32"}
_NUMPY_DTYPE_TO_STR = {
    np.bool_: "bool",
    np.int32: "int32",
    np.float16: "float16",
    np.float32: "float32",
}

_NUMPY_DTYPE_TO_STR.update({np.dtype(k): v for k, v in _NUMPY_DTYPE_TO_STR.items()})

_all_dtypes = [
    "bool",
    "int32",
    "float16",
    "float32",
]

__all__ = list(_all_dtypes) + [
    "dtype",
]
