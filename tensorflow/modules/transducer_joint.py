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

        self.num_classes = num_classes
        self.dim_model = dim_model
        self.activation = activation

        activation = activation.lower()
        if activation == "sigmoid":
            self.activate = tf.keras.activations.sigmoid
        elif activation == "relu":
            self.activate = tf.keras.activations.relu
        elif activation == "tanh":
            self.activate = tf.keras.activations.tanh
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

    def get_config(self):
        conf = super(TransducerJoint, self).get_config()

        conf.update(
            {
                "num_classes": self.num_classes,
                "dim_model": self.dim_model,
                "activation": self.activation,
            }
        )

        return conf

    def call(self, inputs, training=False, **kwargs):
        """

        Args:
            encoder_outputs: [B, T, D_e]
            predictor_outputs: [B, U, D_p]

        """
        encoder_outputs, predictor_outputs = inputs

        f = self.linear_encoder(encoder_outputs)
        f = tf.expand_dims(f, axis=2)  # [B, T, 1, D]
        del encoder_outputs

        g = self.linear_predictor(predictor_outputs)
        g = tf.expand_dims(g, axis=1)  # [B, 1, U, D]
        del predictor_outputs

        joint_inputs = self.activate(f + g)
        del f, g

        joint_outs = self.linear_joint(joint_inputs)
        del joint_inputs

        return joint_outs
