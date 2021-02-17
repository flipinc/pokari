import tensorflow as tf


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def merge_two_last_dims(x):
    b, _, f, c = shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])


def get_rnn(rnn_type: str):
    assert rnn_type in ["lstm", "gru", "rnn"]
    if rnn_type.lower() == "lstm":
        return tf.keras.layers.LSTM
    if rnn_type.lower() == "gru":
        return tf.keras.layers.GRU
    return tf.keras.layers.SimpleRNN


def get_reduced_length(length, reduction_factor):
    return tf.cast(
        tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))),
        dtype=tf.int32,
    )


def find_max_length_prediction_tfarray(tfarray: tf.TensorArray) -> tf.Tensor:
    with tf.name_scope("find_max_length_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = tf.constant(0, dtype=tf.int32)

        def condition(index, _):
            return tf.less(index, total)

        def body(index, max_length):
            prediction = tfarray.read(index)
            length = tf.shape(prediction)[0]
            max_length = tf.where(tf.greater(length, max_length), length, max_length)
            return index + 1, max_length

        index, max_length = tf.while_loop(
            condition, body, loop_vars=[index, max_length], swap_memory=False
        )
        return max_length


def pad_prediction_tfarray(
    tfarray: tf.TensorArray, blank: int or tf.Tensor
) -> tf.TensorArray:
    with tf.name_scope("pad_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = find_max_length_prediction_tfarray(tfarray)

        def condition(index, _):
            return tf.less(index, total)

        def body(index, tfarray):
            prediction = tfarray.read(index)
            prediction = tf.pad(
                prediction,
                paddings=[[0, max_length - tf.shape(prediction)[0]]],
                mode="CONSTANT",
                constant_values=blank,
            )
            tfarray = tfarray.write(index, prediction)
            return index + 1, tfarray

        index, tfarray = tf.while_loop(
            condition, body, loop_vars=[index, tfarray], swap_memory=False
        )
        return tfarray
