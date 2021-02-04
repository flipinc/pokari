from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from modules.lstm import label_collate


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's
        score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a
        packed torch.Tensor behaving in the same manner. dtype must be torch.Long
        in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None


@dataclass
class NBestHypotheses:
    """List of N best hypotheses"""

    n_best_hypotheses: Optional[List[Hypothesis]]


class GreedyInference(nn.Module):
    """A greedy transducer decoder.

    Batch level greedy decoding, performed auto-repressively.

    Args:
        decoder_model: RNNT decoder model.
        joint_model: RNNT joint model.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
            max_symbols_per_step: Optional int. The maximum number of symbols
            that can be added to a sequence in a single time step; if set to
            None then there is no limit.
    """

    def __init__(
        self,
        predictor,
        joint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__()

        self.predictor = predictor
        self.joint = joint

        self._blank_index = blank_index
        self._SOS = blank_index
        self.max_symbols = max_symbols_per_step

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        state: Optional[List[torch.Tensor]],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            state: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as
                "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.predictor.predict(
                    None, state, add_sos=add_sos, batch_size=batch_size
                )

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.predictor.predict(
            label, state, add_sos=add_sos, batch_size=batch_size
        )

    @torch.no_grad()
    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None):
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize
                only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        logits = self.joint.joint(enc, pred)

        if log_normalize is None:
            if not logits.is_cuda:  # Use log softmax only if on CPU
                logits = logits.log_softmax(dim=len(logits.shape) - 1)
        else:
            if log_normalize:
                logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        cache_rnn_state=None,
        mode="normal",
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        predictor_training_state = self.predictor.training
        joint_training_state = self.joint.training

        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            logitlen = encoded_lengths

            self.predictor.eval()
            self.joint.eval()

            with self.predictor.as_frozen(), self.joint.as_frozen():
                inseq = encoder_output  # [B, T, D]
                hypotheses = self._greedy_batch_decode(
                    inseq, logitlen, device=inseq.device
                )

            if mode == "stream":
                return (hypotheses,), cache_rnn_state

            # Pack the hypotheses results
            packed_result = [
                Hypothesis(y_sequence=torch.tensor(sent, dtype=torch.long), score=-1.0)
                for sent in hypotheses
            ]

            del hypotheses

        self.predictor.train(predictor_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,), cache_rnn_state

    @torch.no_grad()
    def _greedy_decode(
        self, x: torch.Tensor, out_len: torch.Tensor, cache_rnn_state=None
    ):
        """Greedy Decode for batch size = 1. Currently not used"""

        # x: [T, 1, D]
        # out_len: [seq_len]

        # Initialize blank state and empty label set
        label = []

        # For timestep t in X_t
        for time_idx in range(out_len):
            # Extract encoder embedding at timestep t
            # f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]
            f = x.narrow(dim=0, start=time_idx, length=1)

            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0

            # While blank is not predicted, or we dont run out of max symbols per
            # timestep
            while not_blank and (
                self.max_symbols is None or symbols_added < self.max_symbols
            ):
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                last_label = self._SOS if label == [] else label[-1]

                # Perform prediction network and joint network steps.
                g, cache_rnn_state_prime = self._pred_step(last_label, cache_rnn_state)
                logp = self._joint_step(f, g, log_normalize=None)[0, 0, 0, :]

                del g

                # torch.max(0) op doesnt exist for FP 16.
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()  # K is the label at timestep t_s in inner loop, s >= 0.

                del logp

                # If blank token is predicted, exit inner loop, move onto next timestep
                if k == self._blank_index:
                    not_blank = False
                else:
                    # Append token to label set, update RNN state.
                    label.append(k)
                    cache_rnn_state = cache_rnn_state_prime

                # Increment token counter.
                symbols_added += 1

        return label, cache_rnn_state

    @torch.no_grad()
    def _greedy_batch_decode(
        self, x: torch.Tensor, out_len: torch.Tensor, device: torch.device
    ):
        # x: [B, T, D]
        # out_len: [B]
        # device: torch.device

        # Initialize state
        hidden = None
        batchsize = x.shape[0]

        # Output string buffer
        label = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full(
            [batchsize, 1],
            fill_value=self._blank_index,
            dtype=torch.long,
            device=device,
        )

        # Mask buffers
        blank_mask = torch.full(
            [batchsize], fill_value=0, dtype=torch.bool, device=device
        )

        # Get max sequence length
        max_out_len = out_len.max()

        for time_idx in range(max_out_len):
            f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask = torch.full_like(blank_mask, 0)

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where
            # current time step T > seq_len
            blank_mask = time_idx >= out_len

            # Start inner loop
            while not_blank and (
                self.max_symbols is None or symbols_added < self.max_symbols
            ):

                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime
                # the state
                if time_idx == 0 and symbols_added == 0:
                    g, hidden_prime = self._pred_step(
                        self._SOS, hidden, batch_size=batchsize
                    )
                else:
                    # Perform batch step prediction of decoder, getting new states and
                    # scores ("g")
                    g, hidden_prime = self._pred_step(
                        last_label, hidden, batch_size=batchsize
                    )

                # Batched joint step - Output = [B, V + 1]
                logp = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]

                if logp.dtype != torch.float32:
                    logp = logp.float()

                # Get index k, of max prob for batch
                v, k = logp.max(1)
                del v, g, logp

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target
                # steps min(max_symbols, U)
                k_is_blank = (k == self._blank_index).to(blank_mask.device)
                # ideally this op should be inplace(bitwise_or_) but trace jit does
                # not support it.
                # blank_mask.bitwise_or_(k_is_blank.to(blank_mask.device))
                blank_mask = blank_mask | k_is_blank

                del k_is_blank

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                result = is_all_one(blank_mask)

                # this does not work for tracing
                # if blank_mask.all():
                if torch.is_nonzero(result):
                    not_blank = False
                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = []
                    if hidden is not None:
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[
                                state_id
                            ][:, blank_indices, :]

                    # Recover prior predicted label for all samples which predicted
                    # blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    last_label = k.clone().view(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample
                    # level loop).
                    for kidx, ki in enumerate(k):
                        # this does not work for tracing
                        # if blank_mask[kidx] == 0:
                        if not torch.is_nonzero(is_zero(blank_mask, kidx)):
                            label[kidx].append(ki)

                    symbols_added += 1

        return label


@torch.jit.script
def is_all_one(tensor: torch.Tensor):
    return torch.tensor(tensor.all() * 1)


@torch.jit.script
def is_zero(tensor: torch.Tensor, idx: int):
    return torch.tensor(tensor[idx] == 0)
