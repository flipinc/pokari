import tensorflow as tf

from utils import shape_list


class TimeReduction(tf.keras.layers.Layer):
    def __init__(self, factor: int, name: str = "TimeReduction", **kwargs):
        super(TimeReduction, self).__init__(name=name, **kwargs)
        self.time_reduction_factor = factor

    def padding(self, time):
        new_time = (
            tf.math.ceil(time / self.time_reduction_factor) * self.time_reduction_factor
        )
        return tf.cast(new_time, dtype=tf.int32) - time

    def call(self, inputs, **kwargs):
        shape = shape_list(inputs)
        outputs = tf.pad(inputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
        outputs = tf.reshape(
            outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor]
        )
        return outputs

    def get_config(self):
        config = super(TimeReduction, self).get_config()
        config.update({"factor": self.time_reduction_factor})
        return config
