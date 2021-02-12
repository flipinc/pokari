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

    @tf.function
    def __call__(self, xs, x_lens):
        x_lens = tf.math.ceil(x_lens / self.hop_length)

        # dither
        if self.dither > 0:
            xs += self.dither * tf.random.normal(tf.shape(xs))

        # preemphasis
        if self.preemph is not None:
            xs = tf.concat(
                (
                    tf.expand_dims(xs[:, 0], axis=1),
                    xs[:, 1:] - self.preemph * xs[:, :-1],
                ),
                axis=1,
            )

        # TODO: is there tensorflow version of torch.cuda.amp.autocast(enabled=False)?
        spectrograms = tf.square(
            tf.abs(
                tf.signal.stft(
                    xs,
                    frame_length=self.win_length,
                    frame_step=self.hop_length,
                    fft_length=self.n_fft,
                    pad_end=True,
                )
            )
        )

        # mel spectrogram
        xs = tf.matmul(self.filterbanks, tf.transpose(spectrograms, [0, 2, 1]))
        del spectrograms

        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                xs = tf.math.log(xs + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                xs = tf.math.log(
                    tf.clip_by_value(xs, clip_value_min=self.log_zero_guard_value)
                )
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # normalize if required
        if self.normalize_type is not None:
            xs = self.normalize_batch(xs, tf.cast(x_lens, tf.int32))

        # TODO: pad to multiple of 8 for efficient tensor core use

        return xs, x_lens

    def normalize_batch(self, xs, x_lens):
        """Normalize audio signal"""

        CONSTANT = 1e-6

        if self.normalize_type == "per_feature":
            mean_list = tf.TensorArray(
                tf.float32, size=xs.shape[0], clear_after_read=True
            )
            std_list = tf.TensorArray(
                tf.float32, size=xs.shape[0], clear_after_read=True
            )
            for i in range(xs.shape[0]):
                mean_list = mean_list.write(
                    i, tf.math.reduce_mean(xs[i, :, : x_lens[i]], axis=1)
                )
                std_list = std_list.write(
                    i, tf.math.reduce_std(xs[i, :, : x_lens[i]], axis=1)
                )
            xs_mean = mean_list.stack()
            xs_std = std_list.stack()
            # make sure xs_std is not zero
            xs_std += CONSTANT
            return (xs - tf.expand_dims(xs_mean, axis=2)) / tf.expand_dims(
                xs_std, axis=2
            )
        else:
            raise NotImplementedError(
                f"{self.normalize_type} is not currently supported."
            )
