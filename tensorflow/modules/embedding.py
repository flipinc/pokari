import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        contraint=None,
        regularizer=None,
        initializer=None,
        **kwargs
    ):
        """
        Keras embedding layer is not supported for TFLite.

        TODO: Add mask zero just like native Embedding impl

        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.contraint = tf.keras.constraints.get(contraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            dtype=tf.float32,
            shape=[self.vocab_size, self.embed_dim],
            initializer=self.initializer,
            trainable=True,
            regularizer=self.regularizer,
            constraint=self.contraint,
        )
        self.built = True

    def call(self, x):
        """

        tf.gather does not work during tflite conversion, so instead using tf.gather_nd
        ref: https://github.com/tensorflow/tensorflow/issues/42410

        """
        x = tf.cast(tf.expand_dims(x, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, x)
