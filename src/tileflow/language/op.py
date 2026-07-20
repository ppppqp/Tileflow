from tileflow.language.ir import OpName, Value, ValueLike, current_builder


def ceildiv(lhs: ValueLike, rhs: ValueLike) -> Value:
    return current_builder().binary(OpName.CEILDIV, lhs, rhs)
