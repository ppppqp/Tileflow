from tileflow.dsl.dtypes import dtype

type DType = dtype | str

# TODO: add PrimExpr support?
type ShapeType = list[int] | tuple[int]
