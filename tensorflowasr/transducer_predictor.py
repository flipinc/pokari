import tensorflow as tf
import tensorflow_addons as tfa


class TransducerPredictor(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim_model: int,
        embed_dim: int,
        random_state_sampling: bool,
        name="transducer_predictor",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.dim_model = dim_model
        self.random_state_sampling = random_state_sampling

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.rnn = LSTMLayerNorm(num_layers, dim_model)

    def get_initial_state(self, batch_size: int, training: bool = True):
        """

        N: number of predictor network layers

        Returns:
            [N, 2, B, D]

        """
        b = batch_size
        states = tf.TensorArray(
            size=self.num_layers, dtype=self.dtype, clear_after_read=True
        )

        if self.random_state_sampling and training:
            for idx in range(self.num_layers):
                h = tf.random.normal(
                    (b, self.dim_model),
                    dtype=self.dtype,
                )
                c = tf.random.normal(
                    (b, self.dim_model),
                    dtype=self.dtype,
                )
                state = tf.stack([h, c], axis=0)
                states = states.write(idx, state)

        else:
            for idx in range(self.num_layers):
                h = tf.zeros(
                    (b, self.dim_model),
                    dtype=self.dtype,
                )
                c = tf.zeros(
                    (b, self.dim_model),
                    dtype=self.dtype,
                )
                state = tf.stack([h, c], axis=0)
                states = states.write(idx, state)

        return states.stack()

    def call(self, targets, target_lens, training=False, **kwargs):
        """

        targets: [B, U]
        target_lens: [B]

        """
        bs = tf.shape(targets)[0]

        targets = self.embed(targets)
        states = self.get_initial_state(bs, training)
        targets, _ = self.rnn(targets, target_lens, states)

        return targets

    def recognize(self, inputs, states):
        """Recognize function for prediction network

        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]

        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](
                outputs, training=False, initial_state=tf.unstack(states[i], axis=0)
            )
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)


class LSTMLayerNorm(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, dim_model: int):
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            lstm_cell = tfa.rnn.LayerNormLSTMCell(units=dim_model)
            lstm_rnn = tf.keras.layers.RNN(
                lstm_cell, return_sequences=True, return_state=True
            )
            self.layers.append(lstm_rnn)

    def call(self, inputs, input_lens, states=None):
        """

        N: number of layers

        Args:
            inputs: [B, U, D]
            states: [N, 2, B, D]

        Returns:
            new_states: [N, 2, B, D]

        """
        new_states = tf.TensorArray(
            self.dtype, size=len(self.layers), clear_after_read=True
        )

        outputs = inputs
        del inputs

        for i, layer in enumerate(self.layers):
            # h and c only accept tuple as an input
            # h, c -> [B, D]
            mask = tf.sequence_mask(input_lens)
            (outputs, h, c) = layer(outputs, (states[i][0], states[i][1]), mask=mask)
            new_states = new_states.write(i, (h, c))

        new_states = new_states.stack()

        return outputs, new_states
