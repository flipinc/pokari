import numpy as np
import six
import tensorflow as tf


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_normalize_signal(signal: tf.Tensor):
    """
    TF Normailize signal to [-1, 1] range
    Args:
        signal: tf.Tensor with shape [None]

    Returns:
        normalized signal with shape [None]
    """
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
    return signal * gain


def tf_normalize_audio_features(audio_feature: tf.Tensor, per_feature=False):
    """
    TF Mean and variance features normalization
    Args:
        audio_feature: tf.Tensor with shape [T, F]

    Returns:
        normalized audio features with shape [T, F]
    """
    axis = 0 if per_feature else None
    mean = tf.reduce_mean(audio_feature, axis=axis)
    std_dev = tf.math.reduce_std(audio_feature, axis=axis) + 1e-9
    return (audio_feature - mean) / std_dev


def tf_preemphasis(signal: tf.Tensor, coeff=0.97):
    """
    TF Pre-emphasis
    Args:
        signal: tf.Tensor with shape [None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        pre-emphasized signal with shape [None]
    """
    if not coeff or coeff <= 0.0:
        return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis=-1)


class SpeechFeaturizer:
    def __init__(self, speech_config: dict):
        """
        We should use TFSpeechFeaturizer for training to avoid differences
        between tf and librosa when converting to tflite in post-training stage
        speech_config = {
            "sample_rate": int,
            "frame_ms": int,
            "stride_ms": int,
            "num_feature_bins": int,
            "feature_type": str,
            "delta": bool,
            "delta_delta": bool,
            "pitch": bool,
            "normalize_signal": bool,
            "normalize_feature": bool,
            "normalize_per_feature": bool
        }
        """
        # Samples
        self.sample_rate = speech_config.get("sample_rate", 16000)
        self.frame_length = int(
            self.sample_rate * (speech_config.get("frame_ms", 25) / 1000)
        )
        self.frame_step = int(
            self.sample_rate * (speech_config.get("stride_ms", 10) / 1000)
        )
        # Features
        self.num_feature_bins = speech_config.get("num_feature_bins", 80)
        self.feature_type = speech_config.get("feature_type", "log_mel_spectrogram")
        self.preemphasis = speech_config.get("preemphasis", None)
        # Normalization
        self.normalize_signal = speech_config.get("normalize_signal", True)
        self.normalize_feature = speech_config.get("normalize_feature", True)
        self.normalize_per_feature = speech_config.get("normalize_per_feature", False)

    @property
    def nfft(self) -> int:
        """ Number of FFT """
        return 2 ** (self.frame_length - 1).bit_length()

    @property
    def shape(self) -> list:
        # None for time dimension
        return [None, self.num_feature_bins, 1]

    def stft(self, signal):
        return tf.square(
            tf.abs(
                tf.signal.stft(
                    signal,
                    frame_length=self.frame_length,
                    frame_step=self.frame_step,
                    fft_length=self.nfft,
                    pad_end=True,
                )
            )
        )

    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        if amin <= 0:
            raise ValueError("amin must be strictly positive")

        magnitude = S

        if six.callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(magnitude)
        else:
            ref_value = np.abs(ref)

        log_spec = 10.0 * log10(tf.maximum(amin, magnitude))
        log_spec -= 10.0 * log10(tf.maximum(amin, ref_value))

        if top_db is not None:
            if top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        features = self.tf_extract(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()

    def tf_extract(self, signal: tf.Tensor) -> tf.Tensor:
        """
        Extract speech features from signals (for using in tflite)
        Args:
            signal: tf.Tensor with shape [None]

        Returns:
            features: tf.Tensor with shape [T, F, 1]
        """
        if self.normalize_signal:
            signal = tf_normalize_signal(signal)
        signal = tf_preemphasis(signal, self.preemphasis)

        features = self.compute_log_mel_spectrogram(signal)

        features = tf.expand_dims(features, axis=-1)

        if self.normalize_feature:
            features = tf_normalize_audio_features(
                features, per_feature=self.normalize_per_feature
            )

        return features

    def compute_log_mel_spectrogram(self, signal):
        spectrogram = self.stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        return self.power_to_db(mel_spectrogram)
