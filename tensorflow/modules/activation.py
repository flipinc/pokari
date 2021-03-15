import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(self, axis=-1, name="glu", **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, x, **kwargs):
        a, b = tf.split(x, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)
