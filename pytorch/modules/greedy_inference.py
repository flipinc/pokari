from typing import Optional, Tuple

import torch
import torch.nn as nn


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
        label: Optional[torch.Tensor],
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ):
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

            # output: [B, 1, K]
            return self.predictor.predict(
                label, state, add_sos=add_sos, batch_size=batch_size
            )

        else:
            return self.predictor.predict(
                None, state, add_sos=add_sos, batch_size=batch_size
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
        cache_rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode: str = "full_context",
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
        with torch.no_grad():
            # TODO: need to set joint and predictor to eval mode
            # currently, torchscript gives error when calling eval() or training(False)

            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            logitlen = encoded_lengths

            inseq = encoder_output  # [B, T, D]
            hypotheses, cache_rnn_state = self._greedy_batch_decode(
                inseq, logitlen, states=cache_rnn_state, device=inseq.device
            )

            # Note: following code will not work for TorchScript
            hypotheses_list = [
                torch.tensor(sent, dtype=torch.long) for sent in hypotheses
            ]

        return hypotheses_list, cache_rnn_state

    @torch.no_grad()
    def _greedy_batch_decode(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        states: torch.Tensor = None,
        device: torch.device = None,
    ):
        # x: [B, T, D]
        # out_len: [B]
        # device: torch.device

        # Initialize state
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = states
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
        max_out_len = int(out_len.max())

        for time_idx in range(max_out_len):
            print(time_idx)
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
                        None, hidden, batch_size=batchsize
                    )
                else:
                    # Perform batch step prediction of decoder, getting new states and
                    # scores ("g")
                    g, hidden_prime = self._pred_step(
                        last_label, hidden, batch_size=batchsize
                    )

                # Batched joint step - Output = [B, V + 1]
                joint_out = self._joint_step(f, g, log_normalize=None)
                logp = joint_out[:, 0, 0, :]

                if logp.dtype != torch.float32:
                    logp = logp.float()

                # Get index k, of max prob for batch
                _, k = logp.max(1)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target
                # steps min(max_symbols, U)
                k_is_blank = (k == self._blank_index).to(blank_mask.device)
                # bitwise_or is not supported in ONNX
                blank_mask.bitwise_or_(k_is_blank)

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False
                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = torch.empty(0).to(dtype=torch.long)
                    if hidden is not None:
                        # Note: as_tuple=False does not work for torchscript
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
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)

                    symbols_added += 1

        return label, hidden
