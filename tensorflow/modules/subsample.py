import math

import tensorflow as tf


class VggSubsample(tf.keras.layers.Layer):
    """Causal Vgg subsampling introduced in https://arxiv.org/pdf/1910.12977.pdf

    Args:
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        activation=tf.keras.activations.relu,
    ):
        super().__init__()

        self.sampling_num = int(math.log(subsampling_factor, 2))

        self.kernel_size = 3
        self.left_padding = self.kernel_size - 1

        self.pool_padding = 0
        self.pool_stride = 2
        self.pool_kernel_size = 2

        self.conv = tf.keras.Sequential()
        for i in range(self.sampling_num):
            self.conv.add(
                tf.keras.layers.Conv2D(
                    filters=conv_channels,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="valid"  # first padding is added manually
                    if i == 0
                    else "same",
                )
            )
            self.conv.add(tf.keras.layers.Activation(activation))
            self.conv.add(
                tf.keras.layers.Conv2D(
                    filters=conv_channels,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                )
            )
            self.conv.add(tf.keras.layers.Activation(activation))
            self.conv.add(
                tf.keras.layers.MaxPool2D(
                    pool_size=self.pool_kernel_size,
                    strides=self.pool_stride,
                    # according to  pytorch implementation, this should be "valid"
                    # but, length will be incorrect
                    padding="same",
                )
            )

        in_length = feat_in
        for i in range(self.sampling_num):
            out_length = self.calc_length(int(in_length))
            in_length = out_length

        self.out = tf.keras.Sequential()
        self.out.add(tf.keras.layers.Input((None, conv_channels * out_length)))
        self.out.add(tf.keras.layers.Dense(feat_out))

    def calc_length(self, in_length: int):
        return math.ceil(
            (in_length + (2 * self.pool_padding) - (self.pool_kernel_size - 1) - 1)
            / self.pool_stride
            + 1
        )

    def calc_tensor_length(self, in_length: tf.Tensor):
        return tf.cast(
            tf.math.ceil(
                (in_length + (2 * self.pool_padding) - (self.pool_kernel_size - 1) - 1)
                / self.pool_stride
                + 1
            ),
            tf.int32,
        )

    def call(self, audio_signals: tf.Tensor, audio_lens: tf.Tensor):
        """
        Args:
            audio_signals (torch.Tensor): [B, Tmax, D]

        Returns:
            [B, Tmax, D]
        """
        # 1. add padding to make this causal convolution
        padding = tf.constant([[0, 0], [1, 1], [self.left_padding, 0]])
        audio_signals = tf.pad(audio_signals, padding)

        # 2. forward
        audio_signals = tf.expand_dims(audio_signals, axis=-1)
        audio_signals = self.conv(audio_signals)
        b = tf.shape(audio_signals)[0]
        t = tf.shape(audio_signals)[1]
        f = tf.shape(audio_signals)[2]
        c = tf.shape(audio_signals)[3]
        audio_signals = tf.reshape(audio_signals, [b, t, c * f])
        audio_signals = self.out(audio_signals)

        # 3. calculate new length
        # TODO: improve the performance of length calculation
        for i in range(self.sampling_num):
            audio_lens = tf.map_fn(self.calc_tensor_length, audio_lens)

        return audio_signals, audio_lens


def stack_subsample(
    audio_signals: tf.Tensor, audio_lens: tf.Tensor, subsampling_factor: int
):
    bs = tf.shape(audio_signals)[0]
    t = tf.shape(audio_signals)[1]
    f = tf.shape(audio_signals)[2]
    t_new = tf.math.ceil(t / subsampling_factor)
    audio_signals = tf.reshape(audio_signals, [bs, t_new, f * subsampling_factor])
    audio_lens = tf.math.ceil(audio_lens / subsampling_factor)
    return audio_signals, audio_lens