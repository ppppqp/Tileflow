from tileflow.typing import ShapeType, DType


class TensorProxy:
    def __call__(self, shape: ShapeType, dtype: DType = "float32"):
        pass


Tensor = TensorProxy()
