import math

import librosa
import tensorflow as tf


class AudioToMelSpectrogramPreprocessor:
    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        normalize_type="per_feature",
        n_fft=None,
        preemph=0.97,
        n_mels=80,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=8,
        max_duration=16.7,
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
        self.n_mels = n_mels
        self.dither = dither
        self.preemph = preemph
        self.mag_power = mag_power
        self.log = log
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.normalize_type = normalize_type

        self.win_length = num_window_size
        self.hop_length = num_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.filterbanks = tf.expand_dims(
            librosa.filters.mel(
                self.sample_rate,
                self.n_fft,
                n_mels=self.n_mels,
                fmin=0.0,
                fmax=(self.sample_rate / 2),
            ),
            axis=0,
        )

    def __call__(self, audio_signals: tf.Tensor, audio_lens: tf.Tensor):
        x = audio_signals
        del audio_signals

        audio_lens = tf.math.ceil(audio_lens / self.hop_length)

        # dither
        if self.dither > 0:
            x += self.dither * tf.random.normal(tf.shape(x))

        # preemphasis
        if self.preemph is not None:
            x = tf.concat(
                (
                    tf.expand_dims(x[:, 0], axis=1),
                    x[:, 1:] - self.preemph * x[:, :-1],
                ),
                axis=1,
            )

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

        # mel spectrogram
        x = tf.matmul(self.filterbanks, tf.transpose(spectrograms, (0, 2, 1)))
        del spectrograms

        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = tf.math.log(x + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                x = tf.math.log(
                    tf.clip_by_value(x, clip_value_min=self.log_zero_guard_value)
                )
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # normalize if required
        if self.normalize_type is not None:
            x = self.normalize_batch(x, tf.cast(audio_lens, tf.int32))

        # TODO: pad to multiple of 8 for efficient tensor core use

        return x, audio_lens

    def normalize_batch(self, x: tf.Tensor, audio_lens: tf.Tensor):
        """Normalize audio signal"""

        CONSTANT = 1e-6

        if self.normalize_type == "per_feature":
            mean_list = tf.TensorArray(
                tf.float32, size=tf.shape(x)[0], clear_after_read=True
            )
            std_list = tf.TensorArray(
                tf.float32, size=tf.shape(x)[0], clear_after_read=True
            )
            for i in range(tf.shape(x)[0]):
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
