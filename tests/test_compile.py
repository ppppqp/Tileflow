import tileflow as tf


@tf.kernel
def add(
    a: tf.TensorType("f32", rank=1),
    b: tf.TensorType("f32", rank=1),
    c: tf.TensorType("f32", rank=1),
):
    i = tf.program_id(0)
    c[i] = a[i] + b[i]


def test_compile_emits_mlir_and_pass_metadata():
    compiled = tf.compile(add)

    assert compiled.name == "add"
    assert "func.func @add" in compiled.mlir
    assert "tileflow.load" in compiled.mlir
    assert "tileflow.store" in compiled.mlir
    assert set(compiled.layouts) == {"a", "b", "c"}
    assert [stage.description for stage in compiled.pipeline] == [
        "global memory read",
        "compute",
        "global memory write",
    ]

