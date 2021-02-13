from typing import Optional, Tuple

import torch
import torch.nn as nn

from modules.lstm import label_collate, rnn


class TransducerPredictor(nn.Module):
    """Transducer Prediction Network comprised of a stateful LSTM model.

    Args:
        dim_model: int specifying the hidden dimension of the prediction net.

        num_layers: int specifying the number of rnn layers.

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

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set
            to None. Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides
            normal-distribution sampled state tensors instead of zero
            tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](
                https://arxiv.org/abs/1910.11455
            )
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        dim_model: int,
        : int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        t_max: int = None,
        forget_gate_bias: int = 1.0,
        dropout: int = 0.0,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.blank_idx = vocab_size

        self.random_state_sampling = random_state_sampling

        self.embed = torch.nn.Embedding(
            vocab_size + 1, embed_dim, padding_idx=self.blank_idx
        )

        self.rnn = rnn(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=num_layers,
            norm=normalization_mode,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
            dropout=dropout,
        )

        # torchscript does not support next(self.parameters()). this is
        # a workaround
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self,
        targets,
        target_lens,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # y: (B, U)
        y = label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        g, _ = self.predict(y, state=states, add_sos=True)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_lens

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        device = self.dummy_param.device
        dtype = self.dummy_param.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            # torchscript does not allow y.device != device
            if not y.device == device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.embed(y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.dim_model), device=device, dtype=dtype)

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
        g, hid = self.rnn(y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        # torchscript does not support del with more than one keyword
        del y
        del start
        del state
        return g, hid

    def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            state = (
                torch.randn(
                    self.num_layers,
                    batch,
                    self.dim_model,
                    dtype=y.dtype,
                    device=y.device,
                ),
                torch.randn(
                    self.num_layers,
                    batch,
                    self.dim_model,
                    dtype=y.dtype,
                    device=y.device,
                ),
            )

        else:
            state = (
                torch.zeros(
                    self.num_layers,
                    batch,
                    self.dim_model,
                    dtype=y.dtype,
                    device=y.device,
                ),
                torch.zeros(
                    self.num_layers,
                    batch,
                    self.dim_model,
                    dtype=y.dtype,
                    device=y.device,
                ),
            )
        return state
