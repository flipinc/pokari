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

    def fn(decode, target):
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

    return tf.numpy_function(fn, inp=[decode, target], Tout=[tf.float32, tf.float32])


def cer(decode: tf.Tensor, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Character Error Rate

    Args:
        decode (tf.Tensor): predicted texts
        target (tf.Tensor): ground truth texts

    Returns:
        Tuple[tf.Tensor]: (edit distances, number of characters)
    """

    # [B, N], [B, M]
    decode = tf.strings.bytes_split(decode)
    target = tf.strings.bytes_split(target)

    # [B]
    distances = tf.edit_distance(
        decode.to_sparse(),
        target.to_sparse(),
        normalize=False,
    )
    lengths = tf.cast(target.row_lengths(axis=1), dtype=tf.float32)

    return tf.reduce_sum(distances), tf.reduce_sum(lengths)


class ErrorRate:
    """Metric for WER and CER

    TODO: should this be `tf.keras.metrics.Metric`?

    """

    def __init__(self, kind: str):
        if kind == "cer":
            self.func = cer
        elif kind == "wer":
            self.func = wer
        else:
            raise ValueError("Available options are `cer` and `wer`.")

        # this is a little temp hack to keep latest value. for why this is
        # required, see a comment on `on_step_end` in models/transducer
        self.value = 100.0

    def __call__(self, decode: tf.Tensor, target: tf.Tensor):
        n, d = self.func(decode, target)
        self.value = tf.math.divide_no_nan(n, d) * 100
        return self.value
