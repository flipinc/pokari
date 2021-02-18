import queue

import tensorflow as tf
from utils.utils import get_reduced_length

from modules.subsample import StackSubsample


class RNNTEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        reduction_indices: list = [0, 1],
        reduction_factors: list = [3, 2],
        num_layers: int = 8,
        dim_model: int = 640,
        num_units: int = 2048,
        name: str = "rnnt_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        reduction_q = queue.Queue()
        for factor in reduction_factors:
            reduction_q.put(factor)

        self.blocks = []
        self.time_reduction_factor = 1
        for i in range(num_layers):
            if i in reduction_indices:
                reduction_factor = reduction_q.get()
                self.time_reduction_factor *= reduction_factor
            else:
                reduction_factor = 0

            self.blocks.append(
                RNNTEncoderBlock(
                    reduction_factor=reduction_factor,
                    dim_model=dim_model,
                    num_units=num_units,
                    name=f"{self.name}_block_{i}",
                )
            )

    def get_initial_state(self):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, P]
        """
        states = []
        for block in self.blocks:
            states.append(
                tf.stack(
                    block.rnn.get_initial_state(tf.zeros([1, 1, 1], dtype=tf.float32)),
                    axis=0,
                )
            )
        return tf.stack(states, axis=0)

    def call(self, x, audio_lens, training=False):
        """

        Args:
            x: [B, T, n_mels]
            audio_lens: [B]

        Returns:
            tf.Tensor: [B, T, D]

        """
        audio_lens = get_reduced_length(audio_lens, self.time_reduction_factor)

        for block in self.blocks:
            x = block(x, training=training)

        return x, audio_lens

    def stream(self, x, states):
        """Stream function for encoder network

        Args:
            inputs (tf.Tensor): shape [1, T, F, C]
            states (tf.Tensor): shape [num_lstms, 1 or 2, 1, P]

        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [num_lstms, 1 or 2, 1, P]
        """
        new_states = []
        for idx, block in enumerate(self.blocks):
            x, new_state = block.stream(x, states=tf.unstack(states[idx], axis=0))
            new_states.append(new_state)

        return x, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.reshape.get_config()
        for block in self.blocks:
            conf.update(block.get_config())
        return conf


class RNNTEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        reduction_factor: int = 0,
        dim_model: int = 640,
        num_units: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if reduction_factor > 0:
            self.reduction = StackSubsample(
                reduction_factor, name=f"{self.name}_stack_subsample"
            )
        else:
            self.reduction = None

        self.rnn = tf.keras.layers.LSTM(
            units=num_units,
            return_sequences=True,
            return_state=True,
        )

        self.ln = tf.keras.layers.LayerNormalization(name=f"{self.name}_ln")

        self.projection = tf.keras.layers.Dense(
            dim_model,
            name=f"{self.name}_projection",
        )

    def call(self, x, training=False, **kwargs):
        if self.reduction is not None:
            x = self.reduction(x)
        (x, _, _) = self.rnn(x)
        x = self.ln(x, training=training)
        x = self.projection(x, training=training)
        return x

    def stream(self, x, states):
        if self.reduction is not None:
            x = self.reduction(x)
        (x, h, c) = self.rnn(x, training=False, initial_state=states)
        x = self.ln(x, training=False)
        x = self.projection(x, training=False)
        return x, tf.stack([h, c], axis=0)

    def get_config(self):
        conf = {}
        if self.reduction is not None:
            conf.update(self.reduction.get_config())
        conf.update(self.rnn.get_config())
        if self.ln is not None:
            conf.update(self.ln.get_config())
        conf.update(self.projection.get_config())
        return conf
