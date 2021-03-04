import tensorflow as tf


class JasperEncoder(tf.keras.Model):
    def __init__(
        self,
        dense: bool = False,
        first_block_channels: int = 256,
        first_block_kernels: int = 11,
        first_block_strides: int = 2,
        first_block_dilation: int = 1,
        first_block_dropout: int = 0.2,
        num_sub_blocks: int = 5,
        main_block_channels: list = [256, 384, 512, 640, 768],
        main_block_kernels: list = [11, 13, 17, 21, 25],
        main_block_dropout: list = [0.2, 0.2, 0.2, 0.3, 0.3],
        second_block_channels: int = 896,
        second_block_kernels: int = 1,
        second_block_strides: int = 1,
        second_block_dilation: int = 2,
        second_block_dropout: int = 0.4,
        third_block_channels: int = 1024,
        third_block_kernels: int = 1,
        third_block_strides: int = 1,
        third_block_dilation: int = 1,
        third_block_dropout: int = 0.4,
        kernel_regularizer=None,
        bias_regularizer=None,
        name: str = "jasper_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.first_block = JasperSubBlock(
            channels=first_block_channels,
            kernels=first_block_kernels,
            strides=first_block_strides,
            dropout=first_block_dropout,
            dilation=first_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_first_block",
        )

        self.main_blocks = [
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                channels=main_block_channels[i],
                kernels=main_block_kernels[i],
                dropout=main_block_dropout[i],
                dense=dense,
                num_residuals=(i + 1) if dense else 1,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}",
            )
            for i in range(len(main_block_channels))
        ]

        self.second_block = JasperSubBlock(
            channels=second_block_channels,
            kernels=second_block_kernels,
            strides=second_block_strides,
            dropout=second_block_dropout,
            dilation=second_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_second_block",
        )

        self.third_block = JasperSubBlock(
            channels=third_block_channels,
            kernels=third_block_kernels,
            strides=third_block_strides,
            dropout=third_block_dropout,
            dilation=third_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_third_block",
        )

        self.reduction_factor = (
            self.first_block.reduction_factor
            * self.second_block.reduction_factor
            * self.third_block.reduction_factor
        )

    def call(self, x, audio_lens, training=False, **kwargs):
        x = self.first_block(x, training=training, **kwargs)

        residuals = []
        for main_block in self.main_blocks:
            x, residuals = main_block([x, residuals], training=training, **kwargs)

        x = self.second_block(x, training=training, **kwargs)
        x = self.third_block(x, training=training, **kwargs)

        audio_lens = tf.cast(
            tf.math.ceil(
                tf.divide(audio_lens, tf.cast(self.reduction_factor, dtype=tf.int32))
            ),
            dtype=tf.int32,
        )

        return x, audio_lens


class JasperBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_sub_blocks: int = 3,
        num_residuals: int = 1,
        channels: int = 256,
        kernels: int = 11,
        dropout: float = 0.1,
        dense: bool = False,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="jasper_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dense = dense

        self.subblocks = [
            JasperSubBlock(
                channels=channels,
                kernels=kernels,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_sub_block_{i}",
            )
            for i in range(num_sub_blocks - 1)
        ]

        self.subblock_residual = JasperSubBlockResidual(
            channels=channels,
            kernels=kernels,
            dropout=dropout,
            num_residuals=num_residuals,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_sub_block_{num_sub_blocks - 1}",
        )

        self.reduction_factor = 1

    def call(self, inputs, training=False, **kwargs):
        inputs, residuals = inputs

        outputs = inputs
        for subblock in self.subblocks:
            outputs = subblock(outputs, training=training, **kwargs)

        if self.dense:
            residuals.append(inputs)
            outputs = self.subblock_residual(
                [outputs, residuals], training=training, **kwargs
            )
        else:
            outputs = self.subblock_residual(
                [outputs, [inputs]], training=training, **kwargs
            )

        return outputs, residuals


class JasperResidual(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int = 256,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="jasper_residual",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.pointwise_conv1d = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_pointwise_conv1d",
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")

    def call(self, x, training=False, **kwargs):
        x = self.pointwise_conv1d(x, training=training)
        x = self.bn(x, training=training)
        return x


class JasperSubBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int = 256,
        kernels: int = 11,
        strides: int = 1,
        dropout: float = 0.1,
        dilation: int = 1,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="jasper_sub_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.conv1d = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=kernels,
            strides=strides,
            dilation_rate=dilation,
            padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv1d",
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.relu = tf.keras.layers.ReLU(name=f"{self.name}_relu")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{self.name}_dropout")

        self.reduction_factor = strides

    def call(self, x, training=False, **kwargs):
        x = self.conv1d(x, training=training)
        x = self.bn(x, training=training)
        x = self.relu(x, training=training)
        x = self.do(x, training=training)
        return x


class JasperSubBlockResidual(JasperSubBlock):
    def __init__(
        self,
        channels: int = 256,
        kernels: int = 11,
        strides: int = 1,
        dropout: float = 0.1,
        dilation: int = 1,
        num_residuals: int = 1,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="jasper_sub_block_residual",
        **kwargs,
    ):
        super().__init__(
            channels=channels,
            kernels=kernels,
            strides=strides,
            dropout=dropout,
            dilation=dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
            **kwargs,
        )

        self.residuals = [
            JasperResidual(
                channels=channels,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_residual_{i}",
            )
            for i in range(num_residuals)
        ]

        self.add = tf.keras.layers.Add(name=f"{self.name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs, residuals = inputs

        outputs = self.conv1d(outputs, training=training)
        outputs = self.bn(outputs, training=training)

        for i, res in enumerate(residuals):
            res = self.residuals[i](res, training=training, **kwargs)
            outputs = self.add([outputs, res], training=training)

        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)

        return outputs
