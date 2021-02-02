from typing import List, Optional

import torch

from modules.greedy_inference import GreedyInference, Hypothesis, NBestHypotheses


class TransducerDecoder(object):
    """
    Used for performing RNN-T auto-regressive decoding of the Decoder+Joint
    network given the encoder state.

    Args:
        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        vocabulary: The vocabulary (excluding the RNNT blank token) which will
        be used for decoding.
    """

    def __init__(self, predictor, joint, vocabulary):
        super().__init__()

        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        self.decoding = GreedyInference(
            predictor=predictor,
            joint=joint,
            blank_index=self.blank_id,
            max_symbols_per_step=30,
        )

    def generate_hypotheses(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        hidden=None,
        streaming=False,
    ) -> (List[str], Optional[List[List[str]]]):
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
        # Compute hypotheses
        with torch.no_grad():
            hypotheses_list, hidden = self.decoding(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                hidden=hidden,
                streaming=streaming,
            )

            # extract the hypotheses
            hypotheses_list: List[Hypothesis] = hypotheses_list[0]

        prediction_list = hypotheses_list

        if isinstance(prediction_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []
            for nbest_hyp in prediction_list:  # type: NBestHypotheses
                n_hyps = (
                    nbest_hyp.n_best_hypotheses
                )  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(n_hyps)  # type: List[str]
                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            return hypotheses, all_hypotheses
        else:
            hypotheses = self.decode_hypothesis(prediction_list)  # type: List[str]
            return hypotheses, hidden

    def decode_hypothesis(self, hypotheses_list: List[Hypothesis]) -> List[str]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.

        Returns:
            A list of strings.
        """
        hypotheses = []
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            prediction = hypotheses_list[ind].y_sequence

            if type(prediction) != list:
                prediction = prediction.tolist()

            # RNN-T sample level is already preprocessed by implicit CTC decoding
            # Simply remove any blank tokens
            prediction = [p for p in prediction if p != self.blank_id]

            # De-tokenize the integer tokens
            hypothesis = "".join(
                [self.labels_map[c] for c in prediction if c != self.blank_id]
            )
            hypotheses.append(hypothesis)

        return hypotheses
