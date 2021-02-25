import math

import tensorflow as tf
from utils.utils import shape_list


class VggSubsample(tf.keras.layers.Layer):
    """Causal Vgg subsampling introduced in https://arxiv.org/pdf/1910.12977.pdf

    Args:
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.

    TODO: This module is very slow in graph mode thus it is not recommend to use this.
    But works fine in eager mode.

    Note: Conv2D on CPU does not support NCHW format

    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int = 64,
        data_format: str = "channels_first",
        name: str = "vgg_subsample",
    ):
        super().__init__(name=name)

        self.sampling_num = int(math.log(subsampling_factor, 2))

        self.kernel_size = 3
        self.left_padding = self.kernel_size - 1

        self.pool_stride = 2
        self.pool_kernel_size = 2

        self.data_format = data_format

        self.layers = []
        for i in range(self.sampling_num):
            conv1 = tf.keras.layers.Conv2D(
                filters=conv_channels,
                kernel_size=self.kernel_size,
                strides=1,
                padding="valid"  # first padding is added manually
                if i == 0
                else "same",
                activation="relu",
                data_format=data_format,
            )
            conv2 = tf.keras.layers.Conv2D(
                filters=conv_channels,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
                activation="relu",
                data_format=data_format,
            )
            pool = tf.keras.layers.MaxPool2D(
                pool_size=self.pool_kernel_size,
                strides=self.pool_stride,
                padding="same",
                data_format=data_format,
            )

            self.layers.append(
                {
                    "conv1": conv1,
                    "conv2": conv2,
                    "pool": pool,
                }
            )

        self.linear_out = tf.keras.layers.Dense(feat_out)

    def calc_length(self, audio_lens: tf.int32):
        return tf.cast(tf.math.ceil(audio_lens / self.pool_stride), tf.int32)

    def call(self, x: tf.Tensor, audio_lens: tf.int32 = None):
        """
        Args:
            x (torch.Tensor): [B, Tmax, D]

        Returns:
            [B, Tmax, D]
        """
        # 1. add padding to make this causal convolution
        x = tf.pad(x, [[0, 0], [self.left_padding, 0], [1, 1]])

        if self.data_format == "channels_first":
            # 2.1 add channel dimension -> [B, 1, Tmax, D]
            x = tf.expand_dims(x, axis=1)
        else:
            # 2.2 add channel dimension -> [B, Tmax, D, 1]
            x = tf.expand_dims(x, axis=-1)

        # 3. forward
        for layer in self.layers:
            x = layer["conv1"](x)
            x = layer["conv2"](x)
            x = layer["pool"](x)

        if self.data_format == "channels_first":
            x = tf.transpose(x, (0, 2, 3, 1))

        b, t, f, c = shape_list(x)
        x = tf.reshape(x, [b, t, f * c])
        x = self.linear_out(x)

        if audio_lens is not None:
            # 4. calculate new length
            # TODO: improve the performance of length calculation
            for i in tf.range(self.sampling_num):
                audio_lens = tf.map_fn(self.calc_length, audio_lens)

            return x, audio_lens
        else:
            return x


class StackSubsample(tf.keras.layers.Layer):
    def __init__(self, subsampling_factor: int, name: str = "stack_subsample"):
        super().__init__(name=name)

        self.subsampling_factor = subsampling_factor

    def call(self, x: tf.Tensor, audio_lens: tf.int32 = None):
        bs, t_max, f = shape_list(x)

        t_new = tf.cast(tf.math.ceil(t_max / self.subsampling_factor), tf.int32)
        pad_amount = tf.cast(t_new * self.subsampling_factor, tf.int32) - t_max

        x = tf.pad(x, [[0, 0], [0, pad_amount], [0, 0]])
        x = tf.reshape(x, [bs, -1, f * self.subsampling_factor])

        if audio_lens is not None:
            audio_lens = tf.cast(
                tf.math.ceil(audio_lens / self.subsampling_factor), tf.int32
            )
            return x, audio_lens
        else:
            return x
