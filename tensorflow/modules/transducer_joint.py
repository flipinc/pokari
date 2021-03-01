import tensorflow as tf


class TransducerJoint(tf.keras.layers.Layer):
    """

    TODO: Sometimes, encoder/predictor dim is not always equal to linear layer dim in
    joint network. To adapt to these cases, the number of dims in linear layer must
    be configurable
    ref: https://arxiv.org/pdf/2010.14665.pdf,

    """

    def __init__(
        self,
        num_classes: int,
        dim_model: int,
        activation: str = "tanh",
        name="transducer_joint",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        activation = activation.lower()
        if activation == "sigmoid":
            self.activation = tf.keras.activations.sigmoid
        elif activation == "relu":
            self.activation = tf.keras.activations.relu
        elif activation == "tanh":
            self.activation = tf.keras.activations.tanh
        else:
            raise ValueError("Activation must be either 'sigmoid', 'relu' or 'tanh'")

        self.linear_encoder = tf.keras.layers.Dense(
            dim_model,
            name=f"{name}_encoder",
        )
        self.linear_predictor = tf.keras.layers.Dense(
            dim_model,
            name=f"{name}_predictor",
        )

        self.linear_joint = tf.keras.layers.Dense(
            num_classes,
            name=f"{name}_joint",
        )

    def call(self, inputs, training=False, **kwargs):
        """

        encoder_outputs: [B, T, D_e]
        predictor_outputs: [B, U, D_p]


        """
        encoder_outputs, predictor_outputs = inputs

        f = self.linear_encoder(encoder_outputs)
        f = tf.expand_dims(f, axis=2)  # [B, T, 1, D]

        g = self.linear_predictor(predictor_outputs)
        g = tf.expand_dims(g, axis=1)  # [B, 1, U, D]

        joint_inputs = self.activation(f + g)
        del f, g

        joint_outs = self.linear_joint(joint_inputs)

        return joint_outs
