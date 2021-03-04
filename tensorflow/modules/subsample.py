import math

import tensorflow as tf
from utils.utils import shape_list


class VggSubsample(tf.keras.layers.Layer):
    def __init__(
        self,
        subsampling_factor: int,
        feat_out: int,
        conv_channels: int = 64,
        data_format: str = "channels_last",
        kernel_regularizer=None,
        bias_regularizer=None,
        name: str = "vgg_subsample",
    ):
        """Causal Vgg subsampling introduced in https://arxiv.org/pdf/1910.12977.pdf
        Args:
            subsampling_factor (int): subsampling factor which should be a power of 2
            feat_in (int): size of the input features
            feat_out (int): size of the output features
            conv_channels (int): Number of channels for the convolution layers.

        Note: Conv2D on CPU does not support NCHW format
        """
        super().__init__(name=name)

        self.sampling_num = int(math.log(subsampling_factor, 2))

        kernel_size = 3
        self.pool_strides = 2

        self.data_format = data_format

        self.layers = []
        for i in range(self.sampling_num):
            conv1 = tf.keras.layers.Conv2D(
                filters=conv_channels,
                kernel_size=kernel_size,
                strides=1,
                padding="valid",
                activation="relu",
                data_format=data_format,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_{i}_conv1",
            )
            conv2 = tf.keras.layers.Conv2D(
                filters=conv_channels,
                kernel_size=kernel_size,
                strides=1,
                padding="valid",
                activation="relu",
                data_format=data_format,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_{i}_conv2",
            )
            pool = tf.keras.layers.MaxPool2D(
                pool_size=self.pool_strides,
                padding="same",
                data_format=data_format,
                name=f"{name}_{i}_pool",
            )

            self.layers.append(
                {
                    "conv1": conv1,
                    "conv2": conv2,
                    "pool": pool,
                }
            )

        self.linear_out = tf.keras.layers.Dense(
            feat_out,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

    def calc_length(self, audio_lens: tf.int32):
        return tf.cast(tf.math.ceil(audio_lens / self.pool_strides), tf.int32)

    def call(self, x: tf.Tensor, audio_lens: tf.int32, training=False, **kwargs):
        """

        Args:
            x: [B, Tmax, D]

        TODO: Add support for causal convolution. For causal conv to work, manual
        padding is required for EVERY LAYER

        """
        if self.data_format == "channels_first":
            # 1.1 add channel dimension -> [B, 1, Tmax, D]
            x = tf.expand_dims(x, axis=1)
        else:
            # 1.2 add channel dimension -> [B, Tmax, D, 1]
            x = tf.expand_dims(x, axis=-1)

        # 2. causal convolution
        for layer in self.layers:
            x = tf.pad(x, [[0, 0], [2, 0], [1, 1], [0, 0]])  # kernel_size - 1 = 2
            x = layer["conv1"](x, training=training)
            x = tf.pad(x, [[0, 0], [2, 0], [1, 1], [0, 0]])
            x = layer["conv2"](x, training=training)
            x = layer["pool"](x, training=training)

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


class ConvSubsample(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: list or tuple or int = 2,
        kernel_size: int or list or tuple = 3,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="conv_subsample",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        self.reduction_factor = self.conv1.strides[0] + self.conv2.strides[0]

    def call(self, x, audio_lens, training=False, **kwargs):
        x = tf.expand_dims(x, axis=-1)

        x = self.conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x, training=training)
        x = tf.nn.relu(x)

        b, t, f, c = shape_list(x)
        x = tf.reshape(x, [b, t, f * c])

        if audio_lens is not None:
            audio_lens = tf.cast(
                tf.math.ceil(
                    tf.divide(
                        audio_lens, tf.cast(self.reduction_factor, dtype=tf.int32)
                    )
                ),
                dtype=tf.int32,
            )

            return x, audio_lens
        else:
            return x


class StackSubsample(tf.keras.layers.Layer):
    def __init__(self, subsampling_factor: int, name: str = "stack_subsample"):
        super().__init__(name=name)

        self.subsampling_factor = subsampling_factor

    def call(self, x: tf.Tensor, audio_lens: tf.int32 = None):
        """

        Args:
            audio_lens: This is None when in streaming mode.

        """
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
