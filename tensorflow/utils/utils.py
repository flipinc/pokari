import tensorflow as tf


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_reduced_length(length, reduction_factor):
    return tf.cast(
        tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))),
        dtype=tf.int32,
    )
