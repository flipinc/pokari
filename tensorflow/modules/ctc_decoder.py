import tensorflow as tf


class CTCDecoder(tf.keras.layers.Layer):
    """Simple decoder for use with CTC-based models

    refs:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
        https://arxiv.org/pdf/2005.04290.pdf
    """

    def __init__(self, num_classes, name="ctc_decoder"):
        super().__init__(name=name)

        self.conv = tf.keras.layers.Conv1D(
            num_classes,
            kernel_size=1,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name=f"{name}_conv",
        )

    def call(self, encoded_outs):
        """
        Args:
            encoded_outs: [B, T, D]
        Returns:
            tf.Tensor: [B, T, num_classes]
        """
        logits = self.conv(encoded_outs)

        return logits
