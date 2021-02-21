import tensorflow as tf

from modules.embedding import Embedding


class TransducerPredictor(tf.keras.layers.Layer):
    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        dim_model: int,
        embed_dim: int,
        random_state_sampling: bool,
        name="transducer_predictor",
        **kwargs,
    ):
        """

        TODO: tensorflow_addons's LSTMLayerNorm is too slow. putting layernorm layer
            after lstm layer substantially increases speed and accuracy. Why?

        """
        super().__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.dim_model = dim_model
        self.random_state_sampling = random_state_sampling

        self.embed = Embedding(num_classes, embed_dim)

        self.rnns = []
        for i in range(num_layers):
            rnn = tf.keras.layers.LSTM(
                units=dim_model,
                return_sequences=True,
                return_state=True,
                name=f"{name}_lstm_{i}",
            )
            ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}")
            self.rnns.append({"rnn": rnn, "ln": ln})

    def get_initial_state(self, batch_size: int, training: bool = False):
        """

        N: number of predictor network layers

        Returns:
            [N, 2, B, D]

        """
        # returning tf.TensorArray().stack() does not work for tensorflow 2.3
        # during build
        states = []

        if self.random_state_sampling and training:
            for idx in range(self.num_layers):
                h = tf.random.normal(
                    (batch_size, self.dim_model),
                    dtype=self.dtype,
                )
                c = tf.random.normal(
                    (batch_size, self.dim_model),
                    dtype=self.dtype,
                )
                state = tf.stack([h, c], axis=0)
                states.append(state)

        else:
            for idx in range(self.num_layers):
                h = tf.zeros(
                    (batch_size, self.dim_model),
                    dtype=self.dtype,
                )
                c = tf.zeros(
                    (batch_size, self.dim_model),
                    dtype=self.dtype,
                )
                state = tf.stack([h, c], axis=0)
                states.append(state)

        return tf.stack(states, axis=0)

    def call(self, targets, target_lens, training=False):
        """

        Args:
            targets: [B, U]
            target_lens: [B]

        """
        bs = tf.shape(targets)[0]

        x = self.embed(targets)
        states = self.get_initial_state(bs, training)

        for idx, rnn in enumerate(self.rnns):
            mask = tf.sequence_mask(target_lens)
            (x, _, _) = rnn["rnn"](
                x,
                training=training,
                mask=mask,
                initial_state=tf.unstack(states[idx], axis=0),
            )
            x = rnn["ln"](x, training=training)

        return x

    def stream(self, x, states):
        """Stream function for prediction network

        Args:
            x (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]

        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        x = self.embed(x, training=False)

        new_states = []
        for idx, rnn in enumerate(self.rnns):
            (x, h, c) = rnn["rnn"](
                x, training=False, initial_state=tf.unstack(states[idx], axis=0)
            )
            new_states.append(tf.stack([h, c]))
            x = rnn["ln"](x, training=False)

        return x, tf.stack(new_states, axis=0)
