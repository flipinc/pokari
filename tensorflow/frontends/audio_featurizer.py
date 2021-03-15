import math

import numpy as np
import tensorflow as tf


class AudioFeaturizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.02,
        window_stride: float = 0.01,
        stft_type: str = "tensorflow",
        normalize_type: str = "per_feature",
        n_fft: int = None,
        preemph: float = 0.97,
        n_mels: int = 80,
        dither: float = 1e-5,
        pad_to: int = 16,
    ):
        """

        Args:
            normalize_type: `per_feature` is preferrable as its value range become more
            contained compared to `per_signal`.
            pad_to: pad to multiple of pad_to for efficient tensor core use. 16 is the
            best value to pick, given that subsequent encoder subsamples by a factor
            of 4 by default.

        """
        self.sample_rate = sample_rate
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.dither = dither
        self.preemph = preemph
        self.n_mels = n_mels
        self.stft_type = stft_type
        self.normalize_type = normalize_type
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.pad_to = pad_to

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        features = self(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()

    def __call__(self, audio_signals, audio_lens):
        """

        Returns:
            tf.Tensor: batch audio features [B, T, n_mels]
            tf.int32: batch audio lengths [B]
        """
        x = audio_signals
        del audio_signals

        audio_lens = tf.cast(tf.math.ceil(audio_lens / self.hop_length), tf.int32)

        # TODO: dither is not supported because tflite does not support random.normal
        # if self.dither > 0:
        #     # align dtype with x for amp (float16)
        #     x += self.dither * tf.random.normal(tf.shape(x), dtype=x.dtype)

        # preemph
        if self.preemph is not None:
            x = tf.concat(
                (
                    tf.expand_dims(x[:, 0], axis=1),
                    x[:, 1:] - self.preemph * x[:, :-1],
                ),
                axis=1,
            )

        # [B, T, nfft/2 + 1]
        if self.stft_type == "tensorflow":
            mxp_enabled = False
            if x.dtype != tf.float32:
                mxp_enabled = True
                x = tf.cast(x, tf.float32)

            # stft can only be run in tf.float32
            stfts = tf.signal.stft(
                x,
                frame_length=self.win_length,
                frame_step=self.hop_length,
                fft_length=self.n_fft,
                pad_end=True,
            )

            if mxp_enabled:
                stfts = tf.cast(stfts, tf.float16)

            # stft returns real & imag, so convert to magnitude
            # x1 for energy spectrogram, x2 for power spectrum
            spectrograms = tf.square(tf.abs(stfts))
        elif self.stft_type == "librosa":
            spectrograms = tf.map_fn(self.librosa_stft, x)

        # [nfft/2 + 1, n_mel]
        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=tf.shape(spectrograms)[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
            dtype=spectrograms.dtype,
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
        mask = tf.expand_dims(tf.range(tf.shape(x)[-1]), 0)
        # [B, T_max] >= [B, 1] -> [B, T_max]
        mask = tf.tile(mask, [tf.shape(x)[0], 1]) >= tf.expand_dims(audio_lens, 1)
        zero = tf.cast(0.0, x.dtype)  # for mxp training
        x = tf.where(tf.expand_dims(mask, 1), zero, x)

        if self.pad_to > 0:
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

    def stream(self, audio_signals, audio_lens):
        x = audio_signals
        del audio_signals

        if self.preemph is not None:
            x = tf.concat(
                (
                    tf.expand_dims(x[:, 0], axis=1),
                    x[:, 1:] - self.preemph * x[:, :-1],
                ),
                axis=1,
            )

        stfts = tf.signal.stft(
            x,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            pad_end=True,
        )

        spectrograms = tf.square(tf.abs(stfts))

        mel_weight = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=tf.shape(spectrograms)[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
        )

        mel_spectrograms = tf.tensordot(spectrograms, mel_weight, 1)
        del spectrograms, mel_weight

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-9)
        del mel_spectrograms

        x = tf.transpose(log_mel_spectrograms, (0, 2, 1))
        del log_mel_spectrograms

        # TODO: optimize for streaming
        if self.normalize_type is not None:
            x = self.normalize(x, audio_lens)

        x = tf.transpose(x, (0, 2, 1))

        return x

    def normalize(self, x: tf.Tensor, audio_lens: tf.Tensor):
        """Normalize audio signal"""

        CONSTANT = 1e-6

        bs = tf.shape(x)[0]

        if self.normalize_type == "per_feature":
            # element_shape of TensorArray must be specified for tflite conversion
            # ref: https://github.com/tensorflow/tensorflow/issues/40221

            mean_list = tf.TensorArray(
                x.dtype,
                size=bs,
                clear_after_read=True,
                element_shape=tf.TensorShape((self.n_mels)),
            )
            std_list = tf.TensorArray(
                x.dtype,
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
        elif self.normalize_type == "per_signal":
            new_x = tf.TensorArray(
                x.dtype,
                size=bs,
                clear_after_read=True,
            )

            for i in range(bs):
                mean = tf.math.reduce_mean(x[i])
                std = tf.math.reduce_std(x[i])
                # make sure x_std is not zero
                std += CONSTANT
                new_x = new_x.write(i, (x[i] - mean) / std)

            return new_x.stack()
        else:
            raise NotImplementedError(
                f"{self.normalize_type} is not currently supported."
            )

    def librosa_stft(self, audio_signal):
        """

        Librosa center pads the signal with reflection whereas tensorflow pad from right
        with zeros by default.

        TODO: this function is not tested. Do not use yet.

        """
        signal = tf.pad(
            audio_signal, [[self.n_fft // 2, self.n_fft // 2]], mode="REFLECT"
        )
        window = tf.signal.hann_window(self.win_length, periodic=True)
        left_pad = (self.n_fft - self.win_length) // 2
        right_pad = self.n_fft - self.win_length - left_pad
        window = tf.pad(window, [[left_pad, right_pad]])
        framed_signals = tf.signal.frame(
            signal, frame_length=self.n_fft, frame_step=self.hop_length
        )
        framed_signals *= window

        return tf.square(tf.abs(tf.signal.rfft(framed_signals, [self.n_fft])))
