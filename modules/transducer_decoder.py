from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class TransducerDecoder(nn.Module):
    """
    Used for performing RNN-T auto-regressive decoding of the Decoder+Joint
    network given the encoder state.

    Args:
        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        vocabulary: The vocabulary (excluding the RNNT blank token) which will
        be used for decoding.
    """

    def __init__(self, labels, inference):
        super().__init__()

        self.blank_id = len(labels)
        self.labels_map = dict([(i, labels[i]) for i in range(len(labels))])
        self.inference = inference

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        cache_rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode: str = "full_context",
    ) -> Tuple[List[str], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Decode an encoder output by autoregressive decoding of the Decoder+Joint
        networks.

        Args:
            encoder_output: torch.Tensor of shape [B, D, T].
            encoded_lengths: torch.Tensor containing lengths of the padded encoder
                outputs. Shape [B].

        Returns:
            If `return_best_hypothesis` is set:
                A tuple (hypotheses, None):
                hypotheses - list of Hypothesis (best hypothesis per sample).
                    Look at rnnt_utils.Hypothesis for more information.

            If `return_best_hypothesis` is not set:
                A tuple(hypotheses, all_hypotheses)
                hypotheses - list of Hypothesis (best hypothesis per sample).
                    Look at rnnt_utils.Hypothesis for more information.
                all_hypotheses - list of NBestHypotheses. Each NBestHypotheses further
                    contains a sorted
                    list of all the hypotheses of the model per sample.
                    Look at rnnt_utils.NBestHypotheses for more information.
        """
        with torch.no_grad():
            hypotheses_list, cache_rnn_state = self.inference(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                cache_rnn_state=cache_rnn_state,
                mode=mode,
            )

        hypotheses = self.decode_hypothesis(hypotheses_list)

        return hypotheses, cache_rnn_state

    def decode_hypothesis(self, hypotheses_list: List[torch.Tensor]) -> List[str]:
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
            prediction_wo_blank: List[int] = []
            for idx, p in enumerate(prediction):
                if p != self.blank_id:
                    prediction_wo_blank.append(p.item())

            # De-tokenize the integer tokens
            hypothesis = self.decode_tokens_to_str(prediction_wo_blank)
            hypotheses.append(hypothesis)

        return hypotheses

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

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
