from tileflow.typing import ShapeType, dtype


class TensorProxy:
    def __call__(self, shape: ShapeType, dtype: type[dtype]):
        pass


Tensor = TensorProxy()
