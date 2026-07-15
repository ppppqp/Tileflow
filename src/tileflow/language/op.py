from tileflow.language.ir import Value, ValueLike, current_builder


def ceildiv(lhs: ValueLike, rhs: ValueLike) -> Value:
    return current_builder().binary("ceildiv", lhs, rhs)
