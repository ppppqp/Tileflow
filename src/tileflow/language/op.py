from tileflow.language.ir import Operation


def ceildiv(lhs, rhs):
    # TODO: return mlir op
    return Operation(kind="tileflow.ceildiv", result=None, operands=[lhs, rhs], attrs={})
