import tensorflow as tf
from utils.utils import shape_list

from modules.activation import GLU
from modules.multihead_attention import (MultiHeadAttention,
                                         RelPositionMultiHeadAttention)
from modules.positional_encoding import (PositionalEncoding,
                                         PositionalEncodingConcat)
from modules.subsample import ConvSubsample, StackSubsample, VggSubsample

L2 = tf.keras.regularizers.l2(1e-6)


# TODO: ADD MASK!!!!!!!!!

class ConformerEncoder(tf.keras.Model):
    def __init__(
        self,
        subsampling="conv",
        positional_encoding="sinusoid",
        dmodel=144,
        num_blocks=16,
        mha_type="relmha",
        head_size=36,
        num_heads=4,
        kernel_size=32,
        depth_multiplier=1,
        fc_factor=0.5,
        dropout=0.0,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if subsampling == "vgg":
            self.conv_subsampling = VggSubsample(
                subsampling_factor=4,
                feat_out=144,
            )
        elif subsampling == "conv":
            self.conv_subsampling = ConvSubsample(
                filters=144,
                kernel_size=3,
                strides=2,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_subsampling",
            )
        elif subsampling == "stack":
            self.conv_subsampling = StackSubsample(
                subsampling_factor=4,
            )
        else:
            raise ValueError(f"Unsupported subsampling `{subsampling}`")

        if positional_encoding == "sinusoid":
            self.pe = PositionalEncoding(name=f"{name}_pe")
        elif positional_encoding == "sinusoid_concat":
            self.pe = PositionalEncodingConcat(name=f"{name}_pe")
        elif positional_encoding == "subsampling":
            self.pe = tf.keras.layers.Activation("linear", name=f"{name}_pe")
        else:
            raise ValueError(
                "positional_encoding must be either 'sinusoid' or 'subsampling'"
            )

        self.linear = tf.keras.layers.Dense(
            dmodel,
            name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_block_{i}",
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, x, audio_lens, training=False, mask=None, **kwargs):
        x, audio_lens = self.conv_subsampling(x, audio_lens, training=training)

        x = self.linear(x, training=training)
        pe = self.pe(x)
        x = self.do(x, training=training)

        for block in self.conformer_blocks:
            x = block([x, pe], training=training, mask=mask, **kwargs)

        return x, audio_lens


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        head_size=36,
        num_heads=4,
        mha_type="relmha",
        kernel_size=32,
        depth_multiplier=1,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.ffm1 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_ff_module_1",
        )
        self.mhsam = MHSAModule(
            mha_type=mha_type,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_mhsa_module",
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            depth_multiplier=depth_multiplier,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_conv_module",
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_ff_module_2",
        )
        self.ln = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer,
            name=f"{name}_ln",
        )

    def call(self, inputs, training=False, mask=None, **kwargs):
        inputs, pos = inputs  # pos is positional encoding

        outputs = self.ffm1(inputs, training=training, **kwargs)
        outputs = self.mhsam([outputs, pos], training=training, mask=mask, **kwargs)
        outputs = self.convm(outputs, training=training, **kwargs)
        outputs = self.ffm2(outputs, training=training, **kwargs)
        outputs = self.ln(outputs, training=training)

        return outputs


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        kernel_size=32,
        dropout=0.0,
        depth_multiplier=1,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conv_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=2 * input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.glu = GLU(name=f"{name}_glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=1,
            padding="same",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dw_conv",
        )
        self.bn = tf.keras.layers.BatchNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            name=f"{name}_bn",
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish, name=f"{name}_swish_activation"
        )
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_pw_conv_2",
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)

        B, T, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, 1, E])

        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])

        return outputs


class MHSAModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="mhsa_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )

        if mha_type == "relmha":
            self.mha = RelPositionMultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")

        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(self, inputs, training=False, mask=None, **kwargs):
        inputs, pos = inputs
        outputs = self.ln(inputs, training=training)

        if self.mha_type == "relmha":
            outputs = self.mha(
                [outputs, outputs, outputs, pos], training=training, mask=mask
            )
        else:
            outputs = outputs + pos
            outputs = self.mha(
                [outputs, outputs, outputs], training=training, mask=mask
            )

        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])

        return outputs


class FFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="ff_module",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            name=f"{name}_ln",
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dense_1",
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish, name=f"{name}_swish_activation"
        )
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dense_2",
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])

        return outputs
