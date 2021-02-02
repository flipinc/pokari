import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VggSubsample(nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: https://arxiv.org/pdf/1910.12977.pdf
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
                    padding=0 if i == 0 else 1,
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
            out_length = self.calc_length(int(in_length))
            in_length = out_length

        self.out = torch.nn.Linear(conv_channels * out_length, feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def calc_length(self, in_length):
        return math.ceil(
            (int(in_length) + (2 * self.pool_padding) - (self.pool_kernel_size - 1) - 1)
            / float(self.pool_stride)
            + 1
        )

    def forward(self, x, lengths, cache=None):
        """

        x (torch.Tensor): [B, Tmax, D]
        cache (torch.Tensor): [B, self.left_padding, D] Cached partial audio input of
            previous segment. Only used in streaming inference.

        """
        # 1. add padding to make this causal convolution
        if cache is None:
            x = F.pad(x, (1, 1, self.left_padding, 0))
        else:
            x = torch.cat([cache, x], dim=1)
            x = F.pad(x, (1, 1, 0, 0))

        # 2. forward
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        # 3. calculate new length
        # TODO: improve the performance of length calculation
        new_lengths = lengths
        for i in range(self.sampling_num):
            new_lengths = [self.calc_length(int(length)) for length in new_lengths]

        new_lengths = torch.IntTensor(new_lengths).to(lengths.device)
        return x, new_lengths


def stack_subsample(audio_signal, lengths, stack_length):
    bs, t_max, idim = audio_signal.shape
    t_new = math.ceil(t_max / stack_length)
    audio_signal = audio_signal.contiguous().view(bs, t_new, idim * stack_length)
    lengths = torch.ceil(lengths / stack_length).int()
    return audio_signal, lengths
