import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VggSubsample(nn.Module):
    """Causal Vgg subsampling introduced in https://arxiv.org/pdf/1910.12977.pdf

    Args:
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self, subsampling_factor, feat_in, feat_out, conv_channels, activation=nn.ReLU()
    ):
        super().__init__()

        self.sampling_num = int(math.log(subsampling_factor, 2))

        self.kernel_size = 3
        self.left_padding = self.kernel_size - 1

        self.pool_padding = 0
        self.pool_stride = 2
        self.pool_kernel_size = 2

        in_channels = 1
        layers = []
        for i in range(self.sampling_num):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0 if i == 0 else 1,  # first padding is added manually
                )
            )
            layers.append(activation)
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(activation)
            layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=self.pool_kernel_size,
                    stride=self.pool_stride,
                    padding=self.pool_padding,
                    ceil_mode=True,
                )
            )
            in_channels = conv_channels

        in_length = feat_in
        for i in range(self.sampling_num):
            out_length = self.calc_length(in_length)
            in_length = out_length

        self.out = torch.nn.Linear(conv_channels * out_length, feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def calc_length(self, in_length: int):
        return math.ceil(
            (int(in_length) + (2 * self.pool_padding) - (self.pool_kernel_size - 1) - 1)
            / float(self.pool_stride)
            + 1
        )

    def forward(self, x, lengths):
        """
        Args:
            x (torch.Tensor): [B, Tmax, D]
        """
        # 1. add padding to make this causal convolution
        x = F.pad(x, (1, 1, self.left_padding, 0))

        # 2. forward
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        # 3. calculate new length
        # TODO: improve the performance of length calculation
        for i in range(self.sampling_num):
            for idx, length in enumerate(lengths):
                lengths[idx] = self.calc_length(length)

        return x, lengths


def stack_subsample(
    audio_signals: torch.Tensor, audio_lens: torch.Tensor, subsampling_factor: int
):
    bs, t_max, idim = audio_signals.shape
    t_new = math.ceil(t_max / subsampling_factor)
    audio_signals = audio_signals.contiguous().view(
        bs, t_new, idim * subsampling_factor
    )
    audio_lens = torch.ceil(audio_lens / subsampling_factor).int()
    return audio_signals, audio_lens
