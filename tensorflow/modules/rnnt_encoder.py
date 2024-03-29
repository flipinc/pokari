import queue

import tensorflow as tf

from modules.subsample import StackSubsample


class RNNTEncoder(tf.keras.Model):
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
        """RNNT Encoder from https://arxiv.org/pdf/1811.06621.pdf"""
        super().__init__(name=name, **kwargs)

        self.reduction_indices = reduction_indices
        self.reduction_factors = reduction_factors
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.num_units = num_units

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

    def get_config(self):
        conf = super(RNNTEncoder, self).get_config()

        conf.update(
            {
                "reduction_indices": self.reduction_indices,
                "reduction_factors": self.reduction_factors,
                "num_layers": self.num_layers,
                "dim_model": self.dim_model,
                "num_units": self.num_units,
            }
        )

        return conf

    def get_initial_state(self, batch_size: int = None):
        """Get zeros states

        Args:
            batch_size: This is not needed for this encoder but all exportable encoders
                must have this function.

        TODO: create meta class for exportable encoders

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, D]
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
        audio_lens = tf.cast(
            tf.math.ceil(tf.divide(audio_lens, self.time_reduction_factor)),
            dtype=tf.int32,
        )

        for block in self.blocks:
            x = block(x, training=training)

        return x, audio_lens

    def stream(self, x, states):
        """Stream function for encoder network

        N: a number of layers

        Args:
            x (tf.Tensor): shape [1, T, n_mels]
            states (tf.Tensor): shape [N, 2, 1, D]

        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [N, 2, 1, D]
        """
        new_states = []
        for idx, block in enumerate(self.blocks):
            x, new_state = block.stream(x, states=tf.unstack(states[idx], axis=0))
            new_states.append(new_state)

        return x, tf.stack(new_states, axis=0)


class RNNTEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        reduction_factor: int = 0,
        dim_model: int = 640,
        num_units: int = 2048,
        name="rnnt_encoder_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.reduction_factor = reduction_factor
        self.dim_model = dim_model
        self.num_units = num_units

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

    def get_config(self):
        conf = super(RNNTEncoderBlock, self).get_config()

        conf.update(
            {
                "reduction_factor": self.reduction_factor,
                "dim_model": self.dim_model,
                "num_units": self.num_units,
            }
        )

        return conf

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
