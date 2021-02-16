import tensorflow as tf

from subsample import TimeReduction
from utils import get_reduced_length, get_rnn


class RNNTEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        reductions: dict = {0: 3, 1: 2},
        dmodel: int = 640,
        nlayers: int = 8,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.blocks = [
            RNNTEncoderBlock(
                reduction_factor=reductions.get(
                    i, 0
                ),  # key is index, value is the factor
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}",
            )
            for i in range(nlayers)
        ]

        self.time_reduction_factor = 1
        for i in range(nlayers):
            reduction_factor = reductions.get(i, 0)
            if reduction_factor > 0:
                self.time_reduction_factor *= reduction_factor

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

    def call(self, x, audio_lens, training=False, **kwargs):
        """

        Args:
            x: [B, T, n_mels]
            audio_lens: [B]

        """
        audio_lens = get_reduced_length(audio_lens, self.time_reduction_factor)

        for block in self.blocks:
            x = block(x, training=training, **kwargs)

        return x, audio_lens

    def recognize(self, inputs, states):
        """Recognize function for encoder network

        Args:
            inputs (tf.Tensor): shape [1, T, F, C]
            states (tf.Tensor): shape [num_lstms, 1 or 2, 1, P]

        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [num_lstms, 1 or 2, 1, P]
        """
        outputs = self.reshape(inputs)
        new_states = []
        for i, block in enumerate(self.blocks):
            outputs, block_states = block.recognize(
                outputs, states=tf.unstack(states[i], axis=0)
            )
            new_states.append(block_states)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.reshape.get_config()
        for block in self.blocks:
            conf.update(block.get_config())
        return conf


class RNNTEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        reduction_factor: int = 0,
        dmodel: int = 640,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(RNNTEncoderBlock, self).__init__(**kwargs)

        if reduction_factor > 0:
            self.reduction = TimeReduction(
                reduction_factor, name=f"{self.name}_reduction"
            )
        else:
            self.reduction = None

        RNN = get_rnn(rnn_type)
        self.rnn = RNN(
            units=rnn_units,
            return_sequences=True,
            name=f"{self.name}_{rnn_type}",
            return_state=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        if layer_norm:
            self.ln = tf.keras.layers.LayerNormalization(name=f"{self.name}_ln")
        else:
            self.ln = None

        self.projection = tf.keras.layers.Dense(
            dmodel,
            name=f"{self.name}_projection",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

    def call(self, x, training=False, **kwargs):
        if self.reduction is not None:
            x = self.reduction(x)
        x = self.rnn(x, training=training)
        x = x[0]
        if self.ln is not None:
            x = self.ln(x, training=training)
        x = self.projection(x, training=training)
        return x

    def recognize(self, x, states):
        if self.reduction is not None:
            x = self.reduction(x)
        x = self.rnn(x, training=False, initial_state=states)
        new_states = tf.stack(x[1:], axis=0)
        x = x[0]
        if self.ln is not None:
            x = self.ln(x, training=False)
        x = self.projection(x, training=False)
        return x, new_states

    def get_config(self):
        conf = {}
        if self.reduction is not None:
            conf.update(self.reduction.get_config())
        conf.update(self.rnn.get_config())
        if self.ln is not None:
            conf.update(self.ln.get_config())
        conf.update(self.projection.get_config())
        return conf
