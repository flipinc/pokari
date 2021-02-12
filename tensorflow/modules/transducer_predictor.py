from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from utils.util import label_collate


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

    def call(self, inputs, states=None):
        """

        N: number of layers

        Args:
            inputs: [B, U, D]
            states: [N, B, D]

        """
        new_states = tf.TensorArray(
            self.dtype, size=len(self.layers), clear_after_read=True
        )

        outputs = inputs
        del inputs

        for i, layer in enumerate(self.layers):
            (outputs, h, c) = layer(outputs, states[i])
            new_states = new_states.write(i, (h, c))

        new_states = new_states.stack()

        return outputs, new_states


class TransducerPredictor(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        dim_model: int,
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        t_max: int = None,
        forget_gate_bias: int = 1.0,
        dropout: int = 0.0,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.blank_idx = vocab_size

        self.random_state_sampling = random_state_sampling

        # TODO: set mask_zero. in order to do this, every idx must be incremented by 1
        self.embed = tf.keras.layers.Embedding(vocab_size + 1, embed_dim)

        self.rnn = LSTMLayerNorm(num_layers, dim_model)

    def call(
        self,
        targets,
        states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        training=True,
    ):
        # y: (B, U)
        y = label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        g, _ = self.predict(
            y, state=states, add_sos=True, training=training
        )  # (B, U, D)
        g = tf.transpose(g, (0, 2, 1))  # (B, D, U)

        return g

    def predict(
        self,
        y: Optional[tf.Tensor] = None,
        state: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
        training: bool = True,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed
                to the Embedding. If None, creates a zero tensor of shape [B, 1, H]
                which mimics output of pad-token on Embedding.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the
                state list length is 2. Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set,
                will sample a normal distribution tensor of the above shape.
                Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal"
                token should be prepended to the above "y" tensor. When set,
                output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None,
                then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:
                g: (B, U, H)
                hid: (h, c) where h is the final sequence hidden state and c is the
                final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

            If add_sos is True:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is the
                final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

        """
        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.embed(y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = tf.zeros((B, 1, self.dim_model), dtype=self.dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = tf.zeros((B, 1, H), dtype=y.dtype)
            y = tf.concat([start, y], axis=1)  # (B, U + 1, H)

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and training:
                state = self.initialize_state(y, training)

        # Forward step through RNN
        g, hid = self.rnn(y, state)

        return g, hid

    def initialize_state(self, y, training: bool = True):
        batch = y.shape[0]
        states = []

        if self.random_state_sampling and training:
            for _ in range(self.num_layers):
                state = (
                    tf.random.normal(
                        (batch, self.dim_model),
                        dtype=y.dtype,
                    ),
                    tf.random.normal(
                        (batch, self.dim_model),
                        dtype=y.dtype,
                    ),
                )
                states.append(state)

        else:
            for _ in range(self.num_layers):
                state = (
                    tf.zeros(
                        (batch, self.dim_model),
                        dtype=y.dtype,
                    ),
                    tf.zeros(
                        (batch, self.dim_model),
                        dtype=y.dtype,
                    ),
                )
                states.append(state)

        return states
