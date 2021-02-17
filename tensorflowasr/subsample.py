import math

import tensorflow as tf

from utils import shape_list


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
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
        activation=tf.keras.activations.relu,
        name: str = "vgg_subsample",
    ):
        super().__init__(name=name)

        self.sampling_num = int(math.log(subsampling_factor, 2))

        self.kernel_size = 3
        self.left_padding = self.kernel_size - 1

        self.pool_padding = 0
        self.pool_stride = 2
        self.pool_kernel_size = 2

        self.layers = []
        for i in range(self.sampling_num):
            conv1 = tf.keras.layers.Conv2D(
                filters=conv_channels,
                kernel_size=self.kernel_size,
                strides=1,
                padding="valid"  # first padding is added manually
                if i == 0
                else "same",
                data_format="channels_first",
            )
            act1 = tf.keras.layers.Activation(activation)
            conv2 = tf.keras.layers.Conv2D(
                filters=conv_channels,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
                data_format="channels_first",
            )
            act2 = tf.keras.layers.Activation(activation)
            pool = tf.keras.layers.MaxPool2D(
                pool_size=self.pool_kernel_size,
                strides=self.pool_stride,
                # according to  pytorch implementation, this should be "valid"
                # but, length will be incorrect
                padding="same",
                data_format="channels_first",
            )

            self.layers.append(
                {
                    "conv1": conv1,
                    "act1": act1,
                    "conv2": conv2,
                    "act2": act2,
                    "pool": pool,
                }
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

    def call(self, x: tf.Tensor, audio_lens: tf.Tensor):
        """
        Args:
            x (torch.Tensor): [B, Tmax, D]

        Returns:
            [B, Tmax, D]
        """
        # 1. add padding to make this causal convolution
        x = tf.pad(x, [[0, 0], [1, 1], [self.left_padding, 0]])

        # 2. add channel dimension -> [B, 1, Tmax, D]
        x = tf.expand_dims(x, axis=1)

        # 2. forward
        for layer in self.layers:
            x = layer["conv1"](x)
            x = layer["act1"](x)
            x = layer["conv2"](x)
            x = layer["act2"](x)
            x = layer["pool"](x)

        b, c, t, f = shape_list(x)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [b, t, c * f])
        x = self.out(x)

        # 3. calculate new length
        # TODO: improve the performance of length calculation
        for i in tf.range(self.sampling_num):
            audio_lens = tf.map_fn(self.calc_tensor_length, audio_lens)

        return x, audio_lens


def call(self, inputs, **kwargs):
    shape = shape_list(inputs)
    outputs = tf.pad(inputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
    outputs = tf.reshape(
        outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor]
    )
    return outputs


class StackSubsample(tf.keras.layers.Layer):
    def __init__(self, subsampling_factor: int, name: str = "stack_subsample"):
        super().__init__(name=name)

        self.subsampling_factor = subsampling_factor

    def call(self, x, audio_lens):
        bs, t_max, f = shape_list(x)

        t_new = tf.cast(tf.math.ceil(t_max / self.subsampling_factor), tf.int32)
        pad_amount = tf.cast(t_new * self.subsampling_factor, tf.int32) - t_max
        audio_lens = tf.cast(
            tf.math.ceil(audio_lens / self.subsampling_factor), tf.int32
        )

        x = tf.pad(x, [[0, 0], [0, pad_amount], [0, 0]])
        x = tf.reshape(x, [bs, -1, f * self.subsampling_factor])

        return x, audio_lens
