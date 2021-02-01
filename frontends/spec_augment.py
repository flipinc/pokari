import random

import torch
import torch.nn as nn


class SpectrogramAugmentation(nn.Module):
    """
    Performs time and freq cuts in one of two ways.
    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.
    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.
    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
    """

    def __init__(
        self,
        freq_masks=2,
        time_masks=2,
        freq_width=15,
        time_width=25,
        rect_masks=5,
        rect_time=25,
        rect_freq=15,
    ):
        super().__init__()

        self.spec_cutout = SpecCutout(
            rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq
        )

        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
            )
        else:
            self.spec_augment = lambda x: x

    def forward(self, input_spec):
        augmented_spec = self.spec_cutout(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class SpecCutout(nn.Module):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20):
        super(SpecCutout, self).__init__()

        self._rng = random.Random()

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = self._rng.randint(0, sh[1] - self.rect_freq)
                rect_y = self._rng.randint(0, sh[2] - self.rect_time)

                w_x = self._rng.randint(0, self.rect_time)
                w_y = self._rng.randint(0, self.rect_freq)

                x[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

        return x


class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    def __init__(self, freq_masks=0, time_masks=0, freq_width=10, time_width=10):
        super(SpecAugment, self).__init__()

        self._rng = random.Random()

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError(
                    "If `time_width` is a float value, must be in range [0, 1]"
                )

            self.adaptive_temporal_width = True

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        if self.adaptive_temporal_width:
            time_width = max(1, int(sh[2] * self.time_width))
        else:
            time_width = self.time_width

        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = self._rng.randint(0, sh[1] - self.freq_width)

                w = self._rng.randint(0, self.freq_width)

                x[idx, x_left : x_left + w, :] = 0.0

            for i in range(self.time_masks):
                y_left = self._rng.randint(0, sh[2] - time_width)

                w = self._rng.randint(0, time_width)

                x[idx, :, y_left : y_left + w] = 0.0

        return x
