from typing import Optional

import tensorflow as tf


class TransducerDecoder:
    def __init__(self, labels: list, inference, blank_idx: int = 0):
        super().__init__()

        self.blank_id = blank_idx
        self.labels = labels
        self.inference = inference

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
