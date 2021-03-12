import tensorflow as tf
from utils.utils import shape_list


class SpectrogramAugmentation:
    """
    Zeroes out(cuts) random continuous horisontal or vertical segments of
    the spectrogram as described in SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
    """

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
    ):
        self.freq_mask = FreqencyMask(num_masks=freq_masks, mask_width=freq_width)
        self.time_mask = TimeMask(num_masks=time_masks, mask_width=time_width)

    def __call__(self, audio_signals):
        """
        Args:
            audio_signals: [B, T, D]
        """
        audio_signals = tf.map_fn(self.freq_mask, audio_signals)
        audio_signals = tf.map_fn(self.time_mask, audio_signals)
        return audio_signals


class FreqencyMask:
    def __init__(self, num_masks: int = 1, mask_width: float = 27):
        self.num_masks = num_masks
        self.mask_width = mask_width

    def __call__(self, spectrogram: tf.Tensor):
        """Masking the frequency channels
        Args:
            spectrogram: shape (T, D)
        Returns:
            frequency masked spectrogram
        """
        T, F = shape_list(spectrogram, out_type=tf.int32)
        for _ in range(self.num_masks):
            f = tf.random.uniform([], minval=0, maxval=self.mask_width, dtype=tf.int32)
            # -1 because minval must be larger than maxval. min=0 max=0 is not allowed
            f = tf.minimum(f, F - 1)
            f0 = tf.random.uniform([], minval=0, maxval=(F - f), dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones([T, f0], dtype=spectrogram.dtype),
                    tf.zeros([T, f], dtype=spectrogram.dtype),
                    tf.ones([T, F - f0 - f], dtype=spectrogram.dtype),
                ],
                axis=1,
            )
            spectrogram = spectrogram * mask
        return spectrogram


class TimeMask:
    def __init__(
        self, num_masks: int = 1, mask_width: float = 100, p_upperbound: float = 1.0
    ):
        self.num_masks = num_masks
        self.mask_width = mask_width
        self.p_upperbound = p_upperbound

    def __call__(self, spectrogram: tf.Tensor):
        """Masking the time channel
        Args:
            spectrogram: shape (T, D)
        Returns:
            frequency masked spectrogram
        """
        T, F = shape_list(spectrogram, out_type=tf.int32)
        for _ in range(self.num_masks):
            t = tf.random.uniform([], minval=0, maxval=self.mask_width, dtype=tf.int32)
            upperbound = tf.cast(
                tf.cast(T, dtype=tf.float32) * self.p_upperbound, dtype=tf.int32
            )
            # -1 because minval must be larger than maxval. min=0 max=0 is not allowed
            t = tf.minimum(t, upperbound - 1)
            t0 = tf.random.uniform([], minval=0, maxval=(T - t), dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones([t0, F], dtype=spectrogram.dtype),
                    tf.zeros([t, F], dtype=spectrogram.dtype),
                    tf.ones([T - t0 - t, F], dtype=spectrogram.dtype),
                ],
                axis=0,
            )
            spectrogram = spectrogram * mask
        return spectrogram
