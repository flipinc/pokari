import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def rnn(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    norm: Optional[str] = None,
    forget_gate_bias: Optional[float] = 1.0,
    dropout: Optional[float] = 0.0,
    norm_first_rnn: Optional[bool] = None,
    t_max: Optional[int] = None,
) -> nn.Module:
    """
    Utility function to provide unified interface to common LSTM RNN modules.

    Args:
        input_size: Input dimension.

        hidden_size: Hidden dimension of the RNN.

        num_layers: Number of RNN layers.

        norm: Optional string representing type of normalization to apply to the RNN.
            Supported values are None, batch and layer.

        forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](
                    http://proceedings.mlr.press/v37/jozefowicz15.pdf
                )

        dropout: Optional dropout to apply to end of multi-layered RNN.

        norm_first_rnn: Whether to normalize the first RNN layer.

        t_max: int value, set to None by default. If an int is specified, performs
            Chrono Initialization of the LSTM network, based on the maximum number of
            timesteps `t_max` expected during the course of training.
            Reference:
            [Can recurrent neural networks warp time?](
                https://openreview.net/forum?id=SJcKhk-Ab
            )

    Returns:
        A RNN module
    """

    if norm is None:
        return LSTMDropout(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forget_gate_bias=forget_gate_bias,
            t_max=t_max,
        )

    # TODO: the shape of one of ln_lstm's ouputs => (hy, cy) is wrong
    # should be same as LSTMDropout
    if norm == "layer":
        return torch.jit.script(
            ln_lstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                t_max=t_max,
            )
        )


class LSTMDropout(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float],
        forget_gate_bias: Optional[float],
        t_max: Optional[int] = None,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.
        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.

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

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LSTMDropout, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        if t_max is not None:
            # apply chrono init
            for name, v in self.lstm.named_parameters():
                if "bias" in name:
                    p = getattr(self.lstm, name)
                    n = p.nelement()
                    hidden_size = n // 4
                    p.data.fill_(0)
                    p.data[hidden_size : 2 * hidden_size] = torch.log(
                        torch.nn.init.uniform_(p.data[0:hidden_size], 1, t_max - 1)
                    )
                    # forget gate biases = log(uniform(1, Tmax-1))
                    p.data[0:hidden_size] = -p.data[hidden_size : 2 * hidden_size]
                    # input gate biases = -(forget gate biases)

        elif forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(0)

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)

        if self.dropout:
            x = self.dropout(x)

        return x, h


def ln_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: Optional[float],
    forget_gate_bias: Optional[float],
    t_max: Optional[int],
) -> torch.nn.Module:
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""
    # The following are not implemented.
    if dropout is not None and dropout != 0.0:
        raise ValueError("`dropout` not supported with LayerNormLSTM")

    if t_max is not None:
        raise ValueError("LayerNormLSTM does not support chrono init")

    return StackedLSTM(
        num_layers,
        LSTMLayer,
        first_layer_args=[LayerNormLSTMCell, input_size, hidden_size, forget_gate_bias],
        other_layer_args=[
            LayerNormLSTMCell,
            hidden_size,
            hidden_size,
            forget_gate_bias,
        ],
    )


def init_stacked_lstm(
    num_layers: int,
    layer: torch.nn.Module,
    first_layer_args: List,
    other_layer_args: List,
) -> torch.nn.ModuleList:
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return torch.nn.ModuleList(layers)


class StackedLSTM(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer: torch.nn.Module,
        first_layer_args: List,
        other_layer_args: List,
    ):
        super(StackedLSTM, self).__init__()
        self.layers: torch.nn.ModuleList = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    def forward(
        self,
        input: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if states is None:
            temp_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            batch = input.size(1)
            for layer in self.layers:
                temp_states.append(
                    (
                        torch.zeros(
                            batch,
                            layer.cell.hidden_size,
                            dtype=input.dtype,
                            device=input.device,
                        ),
                        torch.zeros(
                            batch,
                            layer.cell.hidden_size,
                            dtype=input.dtype,
                            device=input.device,
                        ),
                    )
                )

            states = temp_states

        output_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states.append(out_state)
            i += 1
        return output, output_states


class LSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LayerNormLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, forget_gate_bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))

        # LayerNorm provide learnable biases
        self.layernorm_i = torch.nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = torch.nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = torch.nn.LayerNorm(hidden_size)

        self.reset_parameters()

        self.layernorm_i.bias.data[hidden_size : 2 * hidden_size].fill_(0.0)
        self.layernorm_h.bias.data[hidden_size : 2 * hidden_size].fill_(
            forget_gate_bias
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels
