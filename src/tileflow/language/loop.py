from tileflow.language.ir import Operation


def Parallel(*extents: int):
    return Operation(name="tileflow.parallel", attrs={"extents": extents})
