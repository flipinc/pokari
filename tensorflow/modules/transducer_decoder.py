from typing import Optional

import tensorflow as tf


class TransducerDecoder(tf.keras.layers.Layer):
    def __init__(self, labels, inference):
        super().__init__()

        self.blank_id = 0
        # + 1 for blank idx
        self.labels = [""] + labels
        self.inference = inference

    @tf.function
    def __call__(
        self,
        encoder_output: tf.Tensor,
        encoded_lengths: tf.Tensor,
        cache_rnn_state: Optional[tf.Tensor] = None,
        mode: str = "full_context",
    ):
        hypotheses_list, cache_rnn_state = self.inference(
            encoded_outs=encoder_output,
            encoded_lens=encoded_lengths,
            cache_rnn_state=cache_rnn_state,
            mode=mode,
        )

        decoded_hypotheses = tf.gather_nd(
            self.labels, tf.expand_dims(hypotheses_list, axis=-1)
        )

        return decoded_hypotheses, cache_rnn_state
