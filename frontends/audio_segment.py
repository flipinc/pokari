import librosa
import numpy as np
import soundfile as sf


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(
        self, samples, sample_rate, target_sr=None, trim=False, trim_db=60, orig_sr=None
    ):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

        self._orig_sr = orig_sr if orig_sr is not None else sample_rate

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB" % (
            type(self),
            self.num_samples,
            self.sample_rate,
            self.duration,
            self.rms_db,
        )

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype("float32")
        if samples.dtype in np.sctypes["int"]:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes["float"]:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(
        cls,
        audio_file,
        target_sr=None,
        int_values=False,
        offset=0,
        duration=0,
        trim=False,
        orig_sr=None,
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param audio_file: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(audio_file, "r") as f:
            dtype = "int32" if int_values else "float32"
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        samples = samples.transpose()
        return cls(
            samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr
        )

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    @property
    def orig_sr(self):
        return self._orig_sr

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number
        of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false,
        `pad_size`
        zeros will be added only to the end.
        """
        self._samples = np.pad(
            self._samples,
            (pad_size if symmetric else 0, pad_size),
            mode="constant",
        )
