import random

import tensorflow as tf

from utils import shape_list


class SpectrogramAugmentation:
    """
    Zeroes out(cuts) random continuous horisontal or vertical segments of
    the spectrogram as described in SpecAugment (https://arxiv.org/abs/1904.08779).

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
        super().__init__()

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

    @tf.function
    def __call__(self, audio_signals):
        # [B, T, D] -> [B, D, T]
        audio_signals = tf.transpose(audio_signals, (0, 2, 1))

        _, f, t = shape_list(audio_signals)

        if self.adaptive_temporal_width:
            time_width = max(1, int(t * self.time_width))
        else:
            time_width = self.time_width

        def specaug(audio_signal: tf.Tensor):
            for i in range(self.freq_masks):
                x_left = tf.random.uniform([], 0, f - self.freq_width, dtype=tf.int32)

                w = tf.random.uniform([], 0, self.freq_width, dtype=tf.int32)

                mask = tf.concat(
                    [
                        tf.ones([x_left, t], dtype=audio_signal.dtype),
                        tf.zeros([w, t], dtype=audio_signal.dtype),
                        tf.ones([f - w - x_left, t], dtype=audio_signal.dtype),
                    ],
                    axis=0,
                )

                audio_signal = audio_signal * mask

            for i in range(self.time_masks):
                y_left = tf.random.uniform([], 0, t - time_width, dtype=tf.int32)

                w = tf.random.uniform([], 0, time_width, dtype=tf.int32)

                mask = tf.concat(
                    [
                        tf.ones([f, y_left], dtype=audio_signal.dtype),
                        tf.zeros([f, w], dtype=audio_signal.dtype),
                        tf.ones([f, t - w - y_left], dtype=audio_signal.dtype),
                    ],
                    axis=1,
                )

                audio_signal = audio_signal * mask

            return audio_signal

        audio_signals = tf.map_fn(specaug, audio_signals)

        # [B, D, T] -> [B, T, D]
        audio_signals = tf.transpose(audio_signals, (2, 1, 0))

        return audio_signals
