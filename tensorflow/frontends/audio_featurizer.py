import math

import numpy as np
import tensorflow as tf


class STFT(tf.keras.layers.Layer):
    """Short time fourier transform

    TODO: STFT can be only run in float32. In the future, when support for mixed
    precision is added, stft must be separeted so it can run in float32

    """

    def __init__(self, win_length, hop_length, n_fft, name="stft"):
        super().__init__(name=name, dtype=tf.float32)

        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def call(self, x):
        x = tf.signal.stft(
            x,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            pad_end=True,
        )

        return x


class AudioFeaturizer(tf.keras.layers.Layer):
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
        name="audio_featurizer",
    ):
        super().__init__(name=name)

        self.sample_rate = sample_rate
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.dither = dither
        self.preemph = preemph
        self.n_mels = n_mels
        self.normalize_type = normalize_type
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.pad_to = pad_to

        self.stft = STFT(
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        features = self(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()

    def call(
        self, audio_signals, audio_lens, training: bool = False, inference: bool = False
    ):
        """

        Returns:
            tf.Tensor: batch audio features [B, T, n_mels]
            tf.int32: btach audio lengths [B]
        """
        x = audio_signals
        del audio_signals

        audio_lens = tf.cast(tf.math.ceil(audio_lens / self.hop_length), tf.int32)

        # dither
        if self.dither > 0:
            # align dtype with x for amp (float16)
            x += self.dither * tf.random.normal(tf.shape(x), dtype=x.dtype)

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
        stfts = self.stft(x)

        # stft returns real & imag, so convert to magnitude
        # x1 for energy spectrogram, x2 for power spectrum
        spectrograms = tf.square(tf.abs(stfts))

        # [nfft/2, n_mel]
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=tf.shape(spectrograms)[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
        )

        # mel spectrograms -> [B, T, m_mels]
        mel_spectrograms = tf.tensordot(spectrograms, mel_weight, 1)
        del spectrograms, mel_weight

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-9)
        del mel_spectrograms

        # [B, n_mels, T]
        x = tf.transpose(log_mel_spectrograms, (0, 2, 1))
        del log_mel_spectrograms

        # normalize if required
        if self.normalize_type is not None:
            x = self.normalize(x, audio_lens)

        # mask to zero any values beyond audio_lens in batch
        if not inference:
            mask = tf.expand_dims(tf.range(tf.shape(x)[-1]), 0)
            # [B, T_max] >= [B, 1] -> [B, T_max]
            mask = tf.tile(mask, [tf.shape(x)[0], 1]) >= tf.expand_dims(audio_lens, 1)
            x = tf.where(tf.expand_dims(mask, 1), 0.0, x)

        # pad to multiple of pad_to for efficient tensor core use
        if training and self.pad_to > 0:
            pad_amount = tf.shape(x)[-1] % self.pad_to
            if pad_amount != 0:
                x = tf.pad(
                    x,
                    [[0, 0], [0, 0], [0, self.pad_to - pad_amount]],
                    constant_values=0.0,
                )

        # [B, T, n_mels]
        x = tf.transpose(x, (0, 2, 1))

        return x, audio_lens

    def normalize(self, x: tf.Tensor, audio_lens: tf.Tensor):
        """Normalize audio signal"""

        CONSTANT = 1e-6

        bs = tf.shape(x)[0]

        if self.normalize_type == "per_feature":
            # element_shape of TensorArray must be specified for tflite conversion
            # ref: https://github.com/tensorflow/tensorflow/issues/40221

            mean_list = tf.TensorArray(
                tf.float32,
                size=bs,
                clear_after_read=True,
                element_shape=tf.TensorShape((self.n_mels)),
            )
            std_list = tf.TensorArray(
                tf.float32,
                size=bs,
                clear_after_read=True,
                element_shape=tf.TensorShape((self.n_mels)),
            )

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
