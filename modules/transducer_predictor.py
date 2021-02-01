from typing import Any, Dict, List, Optional, Tuple

import contextmanager
import torch
import torch.nn as nn

from modules.greedy_inference import Hypothesis
from modules.lstm import label_collate, rnn


class RNNTDecoder(nn.Module):
    """A Recurrent Neural Network Transducer Decoder /Prediction Network
    (RNN-T Prediction Network). An RNN-T Decoder/Prediction network,
    comprised of a stateful LSTM model.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            pred_rnn_layers: int specifying the number of rnn layers.

            Optionally, it may also contain the following:
            forget_gate_bias: float, set by default to 1.0, which constructs
                a forget gate initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](
                    http://proceedings.mlr.press/v37/jozefowicz15.pdf
                )
            t_max: int value, set to None by default. If an int is specified, performs
                Chrono Initialization of the LSTM network, based on the maximum number
                of timesteps `t_max` expected during the course of training.
                Reference:
                [Can recurrent neural networks warp time?](
                    https://openreview.net/forum?id=SJcKhk-Ab
                )
            dropout: float, set to 0.0 by default. Optional dropout applied at the end
                of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the
            Prediction network, excluding the RNNT blank token.

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set
            to None. Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides
            normal-distribution sampled state tensors instead of zero
            tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](
                https://arxiv.org/abs/1910.11455
            )

        blank_as_pad: bool, set to True by default. When set, will add a token to the
            Embedding layer of this prediction network, and will treat this token as a
            pad token. In essence, the RNNT pad token will be treated as a pad token,
            and the embedding layer will return a zero tensor for this token.

            It is set by default as it enables various batch optimizations required for
            batched beam search. Therefore, it is not recommended to disable this flag.
    """

    def __init__(
        self,
        pred_hidden: int,
        emded_dim: int,
        num_layers: int,
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
    ):
        super().__init__()

        # Required arguments
        self.pred_hidden = pred_hidden
        self.emded_dim = emded_dim
        self.pred_rnn_layers = num_layers
        self.blank_idx = vocab_size

        self.random_state_sampling = random_state_sampling

        self.prediction = self._predict(
            vocab_size=vocab_size,  # add 1 for blank symbol
            pred_embed_dim=self.emded_dim,
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=self.pred_rnn_layers,
            norm=normalization_mode,
        )

    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        g, _ = self.predict(y, state=states, add_sos=True)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, List[torch.Tensor]):
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed
                to the Embedding. If None, creates a zero tensor of shape [B, 1, H]
                which mimics output of pad-token on Embedding.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the
                state list length is 2. Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set,
                will sample a normal distribution tensor of the above shape.
                Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal"
                token should be prepended to the above "y" tensor. When set,
                output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None,
                then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:
                g: (B, U, H)
                hid: (h, c) where h is the final sequence hidden state and c is the
                final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

            If add_sos is True:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is the
                final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)

        """
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state
        return g, hid

    def _predict(
        self, vocab_size, pred_embed_dim, pred_n_hidden, pred_rnn_layers, norm
    ):
        """
        Prepare the trainable parameters of the Prediction Network.

        Args:
            vocab_size: Vocab size (excluding the blank token).
            pred_n_hidden: Hidden size of the RNNs.
            pred_rnn_layers: Number of RNN layers.
            forget_gate_bias: Whether to perform unit forget gate bias.
            t_max: Whether to perform Chrono LSTM init.
            norm: Type of normalization to perform in RNN.
            dropout: Whether to apply dropout to RNN.
        """
        embed = torch.nn.Embedding(
            vocab_size + 1, pred_embed_dim, padding_idx=self.blank_idx
        )

        layers = torch.nn.ModuleDict(
            {
                "embed": embed,
                "dec_rnn": rnn(
                    input_size=pred_n_hidden,
                    hidden_size=pred_n_hidden,
                    num_layers=pred_rnn_layers,
                    norm=norm,
                    forget_gate_bias=1,
                ),
            }
        )
        return layers

    def initialize_state(self, y: torch.Tensor) -> List[torch.Tensor]:
        """
        Initialize the state of the RNN layers, with same dtype and device as input `y`.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            List of torch.Tensor, each of shape [L, B, H], where
                L = Number of RNN layers
                B = Batch size
                H = Hidden size of RNN.
        """
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = [
                torch.randn(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
                torch.randn(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
            ]

        else:
            state = [
                torch.zeros(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
                torch.zeros(
                    self.pred_rnn_layers,
                    batch,
                    self.pred_hidden,
                    dtype=y.dtype,
                    device=y.device,
                ),
            ]
        return state

    def score_hypothesis(
        self, hypothesis: Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """
        Similar to the predict() method, instead this method scores a Hypothesis during
        beam search. Hypothesis is a dataclass representing one
        hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to rnnt_utils.Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last
            token in the Hypothesis. state is a list of RNN states, each of shape
            [L, 1, H]. lm_token is the final integer token of the hypothesis.
        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if (
            len(hypothesis.y_sequence) > 0
            and hypothesis.y_sequence[-1] == self.blank_idx
        ):
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full(
            [1, 1],
            fill_value=hypothesis.y_sequence[-1],
            device=device,
            dtype=torch.long,
        )
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(
                    None, state=None, add_sos=False, batch_size=1
                )  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self,
        hypotheses: List[Hypothesis],
        cache: Dict[Tuple[int], Any],
        batch_states: List[torch.Tensor],
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.
            batch_states: List of torch.Tensor which represent the states of the RNN
                for this batch. Each state is of shape [L, B, H]

        Returns:
            Returns a tuple (b_y, b_states, lm_tokens) such that:
            b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the
                last tokens in the Hypotheses.
            b_state is a list of list of RNN states, each of shape [L, B, H].
                Represented as B x List[states].
            lm_token is a list of the final integer tokens of the hypotheses
                in the batch.
        """
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and
        # the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(
                batch, -1
            )
            dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
            dec_states = self.batch_initialize_states(
                dec_states, [d_state for seq, d_state in process]
            )

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(
            batch_states, [d_state for y_j, d_state in done]
        )

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done])

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor(
            [h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long
        ).view(final_batch)

        return batch_y, batch_states, lm_tokens

    def batch_initialize_states(
        self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]
    ):
        """
        Create batch of decoder states.

       Args:
           batch_states (list): batch of decoder states
              ([L x (B, H)], [L x (B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([L x (1, H)], [L x (1, H)])]

       Returns:
           batch_states (tuple): batch of decoder states
               ([L x (B, H)], [L x (B, H)])
       """
        # LSTM has 2 states
        for layer in range(self.pred_rnn_layers):
            for state_id in range(len(batch_states)):
                batch_states[state_id][layer] = torch.stack(
                    [s[state_id][layer] for s in decoder_states]
                )

        return batch_states

    def batch_select_state(
        self, batch_states: List[torch.Tensor], idx: int
    ) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        state_list = []
        for state_id in range(len(batch_states)):
            states = [
                batch_states[state_id][layer][idx]
                for layer in range(self.pred_rnn_layers)
            ]
            state_list.append(states)

        return state_list

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally
        unfreezes the module.
        """
        self.freeze()

        try:
            yield
        finally:
            self.unfreeze()
