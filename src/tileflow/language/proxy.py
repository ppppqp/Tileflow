from tileflow.typing import ShapeType, dtype


class TensorProxy:
    def __call__(self, shape: ShapeType, dtype: dtype):
        pass


Tensor = TensorProxy()
