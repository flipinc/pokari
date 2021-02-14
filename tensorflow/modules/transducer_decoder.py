from typing import List, Optional, Tuple

import tensorflow as tf


class TransducerDecoder(tf.keras.layers.Layer):
    def __init__(self, labels, inference: tf.keras.layers.Layer):
        self.blank_id = 0
        # + 1 for blank idx
        self.labels_map = dict([(i + 1, labels[i]) for i in range(len(labels))])
        self.inference = inference

    def call(
        self,
        encoder_output: tf.Tensor,
        encoded_lengths: tf.Tensor,
        cache_rnn_state: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        mode: str = "full_context",
    ) -> Tuple[List[str], Optional[Tuple[tf.Tensor, tf.Tensor]]]:
        hypotheses_list, cache_rnn_state = self.inference(
            encoder_output=encoder_output,
            encoded_lengths=encoded_lengths,
            cache_rnn_state=cache_rnn_state,
            mode=mode,
        )

        hypotheses = self.decode_hypothesis(hypotheses_list)

        return hypotheses, cache_rnn_state

    def decode_hypothesis(self, hypotheses_list: List[tf.Tensor]) -> List[str]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of strings.
        """
        hypotheses: List[str] = []
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            prediction = hypotheses_list[ind]

            # RNN-T sample level is already preprocessed by implicit CTC decoding
            # Simply remove any blank tokens
            tokens: List[int] = []
            for p in prediction:
                if p != self.blank_id:
                    tokens.append(p.item())

            # De-tokenize the integer tokens
            hypothesis = self.decode_tokens_to_str(tokens)
            hypotheses.append(hypothesis)

        return hypotheses

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decode a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis: List[str] = []
        for c in tokens:
            if c != self.blank_id:
                hypothesis.append(self.labels_map[c])

        return "".join(hypothesis)
