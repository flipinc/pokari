import tensorflow as tf


class TransducerJoint(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        joint_dim: int = 1024,
        activation: str = "tanh",
        prejoint_linear: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="tranducer_joint",
        **kwargs,
    ):
        super(TransducerJoint, self).__init__(name=name, **kwargs)

        activation = activation.lower()
        if activation == "linear":
            self.activation = tf.keras.activation.linear
        elif activation == "relu":
            self.activation = tf.nn.relu
        elif activation == "tanh":
            self.activation = tf.nn.tanh
        else:
            raise ValueError("activation must be either 'linear', 'relu' or 'tanh'")

        self.prejoint_linear = prejoint_linear

        if self.prejoint_linear:
            self.ffn_enc = tf.keras.layers.Dense(
                joint_dim,
                name=f"{name}_enc",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
            self.ffn_pred = tf.keras.layers.Dense(
                joint_dim,
                use_bias=False,
                name=f"{name}_pred",
                kernel_regularizer=kernel_regularizer,
            )
        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size,
            name=f"{name}_vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        if self.prejoint_linear:
            enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
            pred_out = self.ffn_pred(
                pred_out, training=training
            )  # [B, U, P] => [B, U, V]
        enc_out = tf.expand_dims(enc_out, axis=2)
        pred_out = tf.expand_dims(pred_out, axis=1)
        outputs = self.activation(enc_out + pred_out)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        conf.update(
            {"prejoint_linear": self.prejoint_linear, "activation": self.activation}
        )
        return conf
