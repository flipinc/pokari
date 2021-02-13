import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from librosa.util import pad_center
from scipy.signal import get_window

from frontends.audio_augment import AudioAugmentor
from frontends.audio_segment import AudioSegment


class STFT(nn.Module):
    """From https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        """
        This module implements an STFT using 1D convolution and 1D transpose
        convolutions. This is a bit tricky so there are some cases that probably
        won't work as working out the same sizes before and after in all overlap
        add setups is tough. Right now, this code should work with hop lengths that
        are half the filter length (50% overlap between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between
                frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame
                (if not specified, it equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming,
                blackman, blackmanharris) (default: {'hann'})
        """
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        win_length = win_length if win_length else filter_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        assert filter_length >= win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())

    def forward(self, input_data):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = torch.nn.functional.conv1d(
            input_data, self.forward_basis, stride=self.hop_length, padding=0
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        return magnitude


class STFTExactPad(STFT):
    """Adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)
        self.pad_amount = (self.filter_length - self.hop_length) // 2


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False, orig_sr=None):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            orig_sr=orig_sr,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)


class AudioToMelSpectrogramPreprocessor(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms."""

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        n_mels=80,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=16,
        max_duration=16.7,
        stft_exact_pad=False,
        stft_conv=False,
        pad_value=0,
        mag_power=2.0,
    ):
        super().__init__()
        self.log_zero_guard_value = log_zero_guard_value
        if (
            window_size is None
            or window_stride is None
            or window_size <= 0
            or window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either window_size or "
                "window_stride. Both must be positive ints."
            )

        if window_size:
            num_window_size = int(window_size * sample_rate)
        if window_stride:
            num_window_stride = int(window_stride * sample_rate)

        self.sample_rate = sample_rate
        self.win_length = num_window_size
        self.hop_length = num_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_exact_pad = stft_exact_pad
        self.stft_conv = stft_conv

        # onnx does not support pytorch native stft yet
        # in that case, stft_conv is the alternative
        # https://github.com/pytorch/pytorch/issues/31317
        if stft_conv:
            if stft_exact_pad:
                self.stft = STFTExactPad(
                    self.n_fft, self.hop_length, self.win_length, window
                )
            else:
                self.stft = STFT(self.n_fft, self.hop_length, self.win_length, window)
        else:
            torch_windows = {
                "hann": torch.hann_window,
                "hamming": torch.hamming_window,
                "blackman": torch.blackman_window,
                "bartlett": torch.bartlett_window,
                "none": None,
            }
            window_fn = torch_windows.get(window, None)
            window_tensor = (
                window_fn(self.win_length, periodic=False) if window_fn else None
            )
            self.register_buffer("window", window_tensor)

            self.stft = lambda x: torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=False if stft_exact_pad else True,
                window=self.window.to(dtype=torch.float, device=x.device),
                return_complex=False,
            )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.n_mels = n_mels
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sample_rate, self.n_fft, n_mels=n_mels, fmin=lowfreq, fmax=highfreq
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(
            torch.tensor(max_duration * sample_rate, dtype=torch.float)
        )
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )
        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    "log_zero_guard_type parameter. It must be either a "
                    "number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        if self.stft_conv:
            # TODO: this is a quick-fix. don't know why this is the right implementation
            return torch.floor(seq_len / self.hop_length).to(dtype=torch.long)
        else:
            return torch.ceil(seq_len / self.hop_length).to(dtype=torch.long)

    def normalize_batch(self, x, seq_len, normalize_type: str):
        """Normalize audio signal"""

        CONSTANT = 1e-6

        if normalize_type == "per_feature":
            x_mean = torch.zeros(
                (seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            )
            x_std = torch.zeros(
                (seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            )
            for i in range(x.shape[0]):
                if x[i, :, : seq_len[i]].shape[1] == 1:
                    raise ValueError(
                        "normalize_batch with `per_feature` normalize_type "
                        "received a tensor of length 1. This will result in "
                        "torch.std() returning nan"
                    )
                x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
                x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        elif normalize_type == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, : seq_len[i]].mean()
                x_std[i] = x[i, :, : seq_len[i]].std()
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        else:
            return x

    @torch.no_grad()
    def forward(self, audio_signals, audio_lens):
        x = audio_signals
        del audio_signals

        audio_lens = self.get_seq_len(audio_lens.float())

        # adjust audio length for native stft so it is same as audio_lens
        if self.stft_exact_pad and not self.stft_conv:
            p = (self.n_fft - self.hop_length) // 2
            x = torch.nn.functional.pad(x.unsqueeze(1), (p, p), "reflect").squeeze(1)

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
            )

        # disable autocast to get full range of stft values
        # when compiling to torchscript, comment the following line
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch returns real, imag; so convert to magnitude
        if not self.stft_conv:
            x = torch.sqrt(x.pow(2).sum(-1))

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # normalize if required
        if self.normalize is not None:
            x = self.normalize_batch(x, audio_lens, normalize_type=self.normalize)

        # mask to zero any values beyond audio_lens in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.expand(x.size(0), max_len) >= audio_lens.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).to(dtype=torch.bool, device=x.device), self.pad_value
        )
        del mask
        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = torch.nn.functional.pad(
                    x, (0, self.pad_to - pad_amt), value=float(self.pad_value)
                )

        return x, audio_lens
