import tileflow as tf


@tf.kernel
def vector_add(
    a: tf.TensorType("f32", rank=1),
    b: tf.TensorType("f32", rank=1),
    c: tf.TensorType("f32", rank=1),
):
    i = tf.program_id(0)
    c[i] = a[i] + b[i]


if __name__ == "__main__":
    compiled = tf.compile(vector_add)
    print(compiled.mlir)

