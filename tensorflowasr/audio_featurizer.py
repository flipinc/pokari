import math

import numpy as np
import tensorflow as tf


class AudioFeaturizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.02,
        window_stride: float = 0.01,
        normalize_type: str = "per_feature",
        n_fft: int = None,
        preemph: float = 0.97,
        n_mels: int = 80,
        dither: float = 1e-5,
        # followings are to used in the future
        pad_to: int = 8,
        pad_value: int = 0,
    ):
        self.sample_rate = sample_rate
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.dither = dither
        self.preemph = preemph
        self.n_mels = n_mels
        self.normalize_type = normalize_type
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

    @property
    def shape(self) -> list:
        # None for time dimension
        return [None, self.n_mels, 1]

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        features = self.tf_extract(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()

    def tf_extract(self, audio_signals, audio_lens):
        # COMPAT (short for compatibility)
        # audio_signals = tf.expand_dims(audio_signals, axis=0)

        x = audio_signals
        del audio_signals

        audio_lens = tf.cast(tf.math.ceil(audio_lens / self.hop_length), tf.int32)

        # dither
        if self.dither > 0:
            x += self.dither * tf.random.normal(tf.shape(x))

        # preemph
        if self.preemph is not None:
            x = tf.concat(
                (
                    tf.expand_dims(x[:, 0], axis=1),
                    x[:, 1:] - self.preemph * x[:, :-1],
                ),
                axis=1,
            )

        # [B, T, nfft/2]
        # TODO: is there tensorflow version of torch.cuda.amp.autocast(enabled=False)?
        spectrograms = tf.square(
            tf.abs(
                tf.signal.stft(
                    x,
                    frame_length=self.win_length,
                    frame_step=self.hop_length,
                    fft_length=self.n_fft,
                    pad_end=True,
                )
            )
        )

        # [nfft/2, n_mel]
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=spectrograms.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
        )

        # mel spectrograms -> [B, T, m_mels]
        x = tf.matmul(spectrograms, mel_weight)
        del spectrograms

        # power to decibel
        x = self.power_to_db(x)

        # [B, n_mels, T]
        x = tf.transpose(x, (0, 2, 1))

        # normalize if required
        if self.normalize_type is not None:
            x = self.normalize(x, audio_lens)

        # TODO: pad to multiple of 8 for efficient tensor core use

        # [B, T, n_mels]
        x = tf.transpose(x, (0, 2, 1))

        return x, audio_lens

    def power_to_db(self, magnitude, ref=1.0, amin=1e-10, top_db=80.0):
        """Conversion from power to decibels. Based off librosa's power_to_db."""

        def log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        if amin <= 0:
            raise ValueError("amin must be strictly positive")

        ref_value = np.abs(ref)

        x = 10.0 * log10(tf.maximum(amin, magnitude))
        x -= 10.0 * log10(tf.maximum(amin, ref_value))

        if top_db is not None:
            if top_db < 0:
                raise ValueError("top_db must be non-negative")
            x = tf.maximum(x, tf.reduce_max(x) - top_db)

        return x

    def normalize(self, x: tf.Tensor, audio_lens: tf.Tensor):
        """Normalize audio signal"""

        CONSTANT = 1e-6

        bs = tf.shape(x)[0]

        if self.normalize_type == "per_feature":
            mean_list = tf.TensorArray(tf.float32, size=bs, clear_after_read=True)
            std_list = tf.TensorArray(tf.float32, size=bs, clear_after_read=True)
            for i in range(bs):
                mean_list = mean_list.write(
                    i, tf.math.reduce_mean(x[i, :, : audio_lens[i]], axis=1)
                )
                std_list = std_list.write(
                    i, tf.math.reduce_std(x[i, :, : audio_lens[i]], axis=1)
                )
            x_mean = mean_list.stack()
            x_std = std_list.stack()
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - tf.expand_dims(x_mean, axis=2)) / tf.expand_dims(x_std, axis=2)
        else:
            raise NotImplementedError(
                f"{self.normalize_type} is not currently supported."
            )
