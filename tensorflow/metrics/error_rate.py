from typing import Tuple

import editdistance
import numpy as np
import tensorflow as tf


def bytes_to_string(array: np.ndarray, encoding: str = "utf-8"):
    if array is None:
        return None
    return [transcript.decode(encoding) for transcript in array]


def wer(decode: np.ndarray, target: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    """Word Error Rate

    Args:
        decode (np.ndarray): array of prediction texts
        target (np.ndarray): array of groundtruth texts

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of words)
    """
    decode = bytes_to_string(decode)
    target = bytes_to_string(target)
    dis = 0.0
    length = 0.0
    for dec, tar in zip(decode, target):
        words = set(dec.split() + tar.split())
        word2char = dict(zip(words, range(len(words))))

        new_decode = [chr(word2char[w]) for w in dec.split()]
        new_target = [chr(word2char[w]) for w in tar.split()]

        dis += editdistance.eval("".join(new_decode), "".join(new_target))
        length += len(tar.split())
    return tf.convert_to_tensor(dis, tf.float32), tf.convert_to_tensor(
        length, tf.float32
    )


def cer(decode: np.ndarray, target: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    """Character Error Rate

    Args:
        decode (np.ndarray): array of prediction texts
        target (np.ndarray): array of groundtruth texts

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of characters)
    """
    decode = bytes_to_string(decode)
    target = bytes_to_string(target)
    dis = 0
    length = 0
    for dec, tar in zip(decode, target):
        dis += editdistance.eval(dec, tar)
        length += len(tar)
    return tf.convert_to_tensor(dis, tf.float32), tf.convert_to_tensor(
        length, tf.float32
    )


class ErrorRate(tf.keras.metrics.Metric):
    """Metric for WER and CER"""

    def __init__(self, kind: str, name="error_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name=f"{name}_numerator", initializer="zeros")
        self.denominator = self.add_weight(
            name=f"{name}_denominator", initializer="zeros"
        )

        if kind == "cer":
            self.func = cer
        elif kind == "wer":
            self.func = wer
        else:
            raise ValueError("Available options are `cer` and `wer`.")

    def update_state(self, decode: tf.Tensor, target: tf.Tensor):
        n, d = tf.numpy_function(
            self.func, inp=[decode, target], Tout=[tf.float32, tf.float32]
        )
        self.numerator.assign_add(n)
        self.denominator.assign_add(d)

    def result(self):
        if self.denominator == 0.0:
            return 0.0
        return (self.numerator / self.denominator) * 100

    def reset_states(self):
        self.numerator.assign(0)
        self.denominator.assign(0)
