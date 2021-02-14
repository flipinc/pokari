import tensorflow as tf


class TransducerJoint(tf.keras.layers.Layer):
    def __init__(
        self,
        encoder_hidden: int,
        predictor_hidden: int,
        dim_model: int,
        activation: str,
        vocab_size: int,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._num_classes = vocab_size + 1  # add 1 for blank symbol

        self.linear_encoder = tf.keras.layers.Dense(dim_model)
        self.linear_predictor = tf.keras.layers.Dense(dim_model)

        activation = activation.lower()

        if activation == "relu":
            activation = tf.keras.activations.relu
        elif activation == "sigmoid":
            activation = tf.keras.activations.sigmoid
        elif activation == "tanh":
            activation = tf.keras.activations.tanh

        self.joint_net = tf.keras.Sequential()
        self.joint_net.add(tf.keras.layers.Activation(activation))
        self.joint_net.add(tf.keras.layers.Dense(self._num_classes))

    def call(
        self,
        encoder_outputs: tf.Tensor,
        predictor_outputs: tf.Tensor,
    ) -> tf.Tensor:
        """

        Args:
            encoder_outputs: (B, D, T)
            predictor_outputs: (B, D, U)

        """

        encoder_outputs = tf.transpose(encoder_outputs, [0, 2, 1])  # (B, T, D)
        predictor_outputs = tf.transpose(predictor_outputs, [0, 2, 1])  # (B, U, D)

        out = self.joint(encoder_outputs, predictor_outputs)  # [B, T, U, V + 1]

        return out

    def joint(self, f: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original
            paper.

            The original paper proposes the following steps :
            (enc, dec)
            -> Expand + Concat + Sum [B, T, U, H1+H2]
            -> Forward through joint hidden [B, T, U, H]
            -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and
            joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2)
            -> Sum [B, T, U, H]
            -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        f = self.linear_encoder(f)
        f = tf.expand_dims(f, axis=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.linear_predictor(g)
        g = tf.expand_dims(g, axis=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        del f, g

        res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        return res
