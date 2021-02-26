import typing

import tensorflow as tf
from utils.utils import shape_list

from modules.subsample import ConvSubsample, VggSubsample

# Conformer in https://arxiv.org/pdf/2005.08100.pdf uses L2 for all weights
L2 = tf.keras.regularizers.l2(1e-6)


class ConformerEncoder(tf.keras.Model):
    """Conformer Encoder from https://arxiv.org/pdf/2005.08100.pdf"""

    def __init__(
        self,
        num_blocks=16,
        dim_model=144,
        num_heads=4,
        head_size=36,
        kernel_size=32,
        depth_multiplier=1,
        fc_factor=0.5,
        dropout=0.0,
        subsampling="conv",
        subsampling_dim=144,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_encoder",
    ):
        super().__init__(name=name)

        if subsampling == "vgg":
            self.subsample = VggSubsample(
                subsampling_factor=4,
                feat_out=dim_model,
                conv_channels=subsampling_dim,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_subsampling",
            )
        elif subsampling == "conv":
            self.subsample = ConvSubsample(
                feat_out=dim_model,
                conv_channels=subsampling_dim,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_subsampling",
            )
        else:
            raise ValueError(
                f"{subsampling} is not supported. Must be either `vgg` or `conv`."
            )

        self.pe = PositionalEncoding(name=f"{name}_pe")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.blocks = []
        for i in range(num_blocks):
            block = ConformerBlock(
                dim_model=dim_model,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_block_{i}",
            )
            self.blocks.append(block)

    def create_mask(self, t, t_max):
        mask = tf.ones([t, t], dtype=tf.float32)
        return tf.pad(mask, [[0, t_max - t], [0, t_max - t]], constant_values=0.0)

    def call(
        self,
        audio_features: tf.Tensor,
        audio_lens: tf.int32,
        training=False,
        mask=None,
    ):
        # audio_features: [B, T, D]
        # audio_lens: [B]

        x, audio_lens = self.subsample(audio_features, audio_lens, training=training)

        pe = self.pe(x)
        x = self.do(x, training=training)

        masks = tf.TensorArray(
            tf.float32, size=tf.shape(audio_lens)[0], clear_after_read=True
        )
        for idx in tf.range(tf.shape(audio_lens)[0]):
            mask = self.create_mask(
                tf.cast(audio_lens[idx], tf.int32),
                tf.cast(tf.shape(x)[1], tf.int32),
            )
            masks = masks.write(idx, mask)
        masks = masks.stack()

        for block in self.blocks:
            x = block([x, pe], mask=masks, training=training)

        return x, audio_lens


class PositionalEncoding(tf.keras.layers.Layer):
    def build(self, input_shape):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(max_len, dmodel):
        pos = tf.expand_dims(tf.range(max_len - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2)) / dmodel))

        # Sin cos will be [max_len, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(
            tf.expand_dims(tf.sin(pe[:, 0::2]), -1),
            [[0, 0], [0, 0], [0, 1]],
            mode="CONSTANT",
            constant_values=0,
        )
        sin = tf.reshape(sin, [max_len, dmodel])
        cos = tf.pad(
            tf.expand_dims(tf.cos(pe[:, 1::2]), -1),
            [[0, 0], [0, 0], [1, 0]],
            mode="CONSTANT",
            constant_values=0,
        )
        cos = tf.reshape(cos, [max_len, dmodel])
        # Then add sin and cos, which results in [time, size]
        pe = tf.add(sin, cos)
        return tf.expand_dims(pe, axis=0)  # [1, time, size]

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_model,
        dropout=0.0,
        fc_factor=0.5,
        head_size=36,
        num_heads=4,
        kernel_size=32,
        depth_multiplier=1,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conformer_block",
    ):
        super().__init__(name=name)

        self.ffm1 = FFModule(
            dim_model=dim_model,
            dropout=dropout,
            fc_factor=fc_factor,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_ff_module_1",
        )
        self.mhsam = MHSAModule(
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_mhsa_module",
        )
        self.convm = ConvModule(
            dim_model=dim_model,
            kernel_size=kernel_size,
            dropout=dropout,
            depth_multiplier=depth_multiplier,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_conv_module",
        )
        self.ffm2 = FFModule(
            dim_model=dim_model,
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

    def call(self, inputs, training=False, mask=None):
        x, pos = inputs

        outs = self.ffm1(x, training=training)
        outs = self.mhsam([outs, pos], mask=mask, training=training)
        outs = self.convm(outs, training=training)
        outs = self.ffm2(outs, training=training)
        outs = self.ln(outs, training=training)

        return outs


class GLU(tf.keras.layers.Layer):
    def __init__(self, axis=-1, name="glu_activation"):
        super().__init__(name=name)
        self.axis = axis

    def call(self, x):
        a, b = tf.split(x, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_model,
        kernel_size=32,
        dropout=0.0,
        depth_multiplier=1,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="conv_module",
        **kwargs,
    ):
        super().__init__(name=name)

        self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=2 * dim_model,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_pw_conv_1",
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
            filters=dim_model,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_pw_conv_2",
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, x, training=False):
        outs = self.ln(x, training=training)

        b, t, d = shape_list(outs)
        outs = tf.reshape(outs, [b, t, 1, d])

        outs = self.pw_conv_1(outs, training=training)
        outs = self.glu(outs)
        outs = self.dw_conv(outs, training=training)
        outs = self.bn(outs, training=training)
        outs = self.swish(outs)
        outs = self.pw_conv_2(outs, training=training)

        outs = tf.reshape(outs, [b, t, d])
        outs = self.do(outs, training=training)
        outs = self.res_add([x, outs])

        return outs


class FFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_model,
        dropout=0.0,
        fc_factor=0.5,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="ff_module",
    ):
        super().__init__(name=name)

        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            name=f"{name}_ln",
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * dim_model,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_linear_1",
        )
        self.swish = tf.keras.layers.Activation(
            tf.nn.swish, name=f"{name}_swish_activation"
        )
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            dim_model,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_linear_2",
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, x, training=False):
        outs = self.ln(x, training=training)
        outs = self.ffn1(outs, training=training)
        outs = self.swish(outs)
        outs = self.do1(outs, training=training)
        outs = self.ffn2(outs, training=training)
        outs = self.do2(outs, training=training)
        outs = self.res_add([x, self.fc_factor * outs])

        return outs


class MHSAModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size,
        num_heads,
        dropout=0.0,
        kernel_regularizer=L2,
        bias_regularizer=L2,
        name="mhsa_module",
        **kwargs,
    ):
        super().__init__(name=name)

        self.ln = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            name=f"{name}_ln",
        )

        self.mha = RelPositionMultiHeadAttention(
            head_size=head_size,
            num_heads=num_heads,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_mhsa",
        )

        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(
        self,
        x,
        training=False,
        mask=None,
    ):
        x, pos = x

        outs = self.ln(x, training=training)

        outs = self.mha([outs, outs, outs, pos], mask=mask, training=training)

        outs = self.do(outs, training=training)
        outs = self.res_add([x, outs])

        return outs


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        head_size,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
        kernel_regularizer: typing.Union[str, typing.Callable] = None,
        kernel_constraint: typing.Union[str, typing.Callable] = None,
        bias_initializer: typing.Union[str, typing.Callable] = "zeros",
        bias_regularizer: typing.Union[str, typing.Callable] = None,
        bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._droput_rate = dropout

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )
        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value, training=False):
        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to "
                "the same as the number of elements in 'value'"
            )
        # Linear transformations
        query = tf.einsum("...NI,HIO->...NHO", query, self.query_kernel)
        key = tf.einsum("...MI,HIO->...MHO", key, self.key_kernel)
        value = tf.einsum("...MI,HIO->...MHO", value, self.value_kernel)

        return query, key, value

    def call_attention(self, query, key, value, logits, training=False, mask=None):
        # mask = attention mask with shape [B, Tquery, Tkey] with 1 is for positions
        # we want to attend, 0 for masked
        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.shape[-3] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to "
                    "the number of elements in 'query'"
                )
            if key.shape[-3] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to "
                    "the number of elements in 'key'"
                )
        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def call(self, inputs, training=False, mask=None):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        output, attn_coef = self.call_attention(
            query, key, value, logits, training=training, mask=mask
        )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def build(self, input_shape):
        num_pos_features = input_shape[-1][-1]
        self.pos_kernel = self.add_weight(
            name="pos_kernel",
            shape=[self.num_heads, num_pos_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.pos_bias_u = self.add_weight(
            name="pos_bias_u",
            shape=[self.num_heads, self.head_size],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
        )
        self.pos_bias_v = self.add_weight(
            name="pos_bias_v",
            shape=[self.num_heads, self.head_size],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
        )
        super(RelPositionMultiHeadAttention, self).build(input_shape[:-1])

    def relative_shift(self, x):
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.reshape(x[:, :, 1:, :], x_shape)
        return x

    def call(self, inputs, training=False, mask=None):
        query, key, value, pos = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        pos = tf.einsum("...MI,HIO->...MHO", pos, self.pos_kernel)

        query_with_u = query + self.pos_bias_u
        query_with_v = query + self.pos_bias_v

        logits_with_u = tf.einsum("...NHO,...MHO->...HNM", query_with_u, key)
        logits_with_v = tf.einsum("...NHO,...MHO->...HNM", query_with_v, pos)
        logits_with_v = self.relative_shift(logits_with_v)

        logits = logits_with_u + logits_with_v

        depth = tf.constant(self.head_size, dtype=tf.float32)
        logits /= tf.sqrt(depth)

        output, attn_coef = self.call_attention(
            query, key, value, logits, training=training, mask=mask
        )

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output
