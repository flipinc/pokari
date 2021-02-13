import random

import numpy as np
import tensorflow_io as tfio
from omegaconf import DictConfig, OmegaConf


class SpeedPerturbation(object):
    def __init__(
        self,
        sr,
        min_speed_rate=0.9,
        max_speed_rate=1.1,
        num_rates=5,
        rng=None,
    ):
        """
        Performs Speed Augmentation by re-sampling the data to a different
        sampling rate, which does not preserve pitch.

        Note: This is a very slow operation for online augmentation. If space allows,
        it is preferable to pre-compute and save the files to augment the dataset.

        Args:
            sr: Original sampling rate.
            resample_type: Type of resampling operation that will be performed.
                For better speed using `resampy`'s fast resampling method, use
                `resample_type='kaiser_fast'`. For high-quality resampling, set
                `resample_type='kaiser_best'`. To use `scipy.signal.resample`,
                set `resample_type='fft'` or `resample_type='scipy'`
            min_speed_rate: Minimum sampling rate modifier.
            max_speed_rate: Maximum sampling rate modifier.
            num_rates: Number of discrete rates to allow. Can be a positive or
                negative integer.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed
                rates will be sampled uniformly.

                Note: If a positive integer is provided and the resultant
                discretized range of rates contains the value '1.0', then those
                samples with rate=1.0, will not be augmented at all and simply
                skipped. This is to unnecessary augmentation and increase
                computation time. Effective augmentation chance in such a case
                is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being
                augmented.
            rng: Random seed number.
        """
        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        self._sr = sr
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(
                self._min_rate, self._max_rate, self._num_rates, endpoint=True
            )
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, audio):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = self._rng.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        new_sr = int(self._sr * speed_rate)
        audio = tfio.audio.resample(audio, self._sr, new_sr)

        return audio


perturbation_types = {
    "speed": SpeedPerturbation,
}


def get_augmentations(augmentor: DictConfig):
    augmentor = OmegaConf.to_container(augmentor, resolve=True)

    augmentations = []
    for augment_name, augment_kwargs in augmentor.items():
        prob = augment_kwargs.get("prob", None)

        if prob is None:
            raise KeyError(
                f'Augmentation "{augment_name}" will not be applied as '
                f'keyword argument "prob" was not defined for this augmentation.'
            )
        elif prob < 0.0 or prob > 1.0:
            raise ValueError("`prob` must be a float value between 0 and 1.")
        else:
            _ = augment_kwargs.pop("prob")

            try:
                augmentation = perturbation_types[augment_name](**augment_kwargs)
                augmentations.append([prob, augmentation])
            except KeyError:
                raise KeyError(
                    "Invalid perturbation name. Allowed values : "
                    f"{perturbation_types.keys()}"
                )

    augmenter = AudioAugmentor(perturbations=augmentations)
    return augmenter


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, audio):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                audio = p.perturb(audio)
        return audio

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen
