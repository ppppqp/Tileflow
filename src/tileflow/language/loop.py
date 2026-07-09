from tileflow.language.ir import Operation


def Parallel(*extents: int):
    return Operation(kind="tileflow.parallel", result=None, operands=list(extents), attrs={})
