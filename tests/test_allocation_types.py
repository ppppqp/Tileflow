import tileflow.language as T
from tileflow.compiler.builder import Builder
from tileflow.language.ir import BufferType, OpName, TileType


def test_shared_storage_and_register_tiles_have_distinct_types():
    with Builder("allocation_types") as builder:
        shared = T.alloc_shared((16, 16), T.float16)
        fragment = T.alloc_fragment((16, 16), T.float32)

    assert isinstance(shared.type, BufferType)
    assert shared.type.memory_space == "shared"
    assert isinstance(fragment.type, TileType)
    assert [op.name for op in builder.ir.body.entry.ops] == [
        OpName.ALLOC,
        OpName.TILE_EMPTY,
    ]
