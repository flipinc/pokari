import sys

import numpy as np
import tensorflow as tf


def print_one_line(*args):
    tf.print("\033[K", end="")
    tf.print("\r", *args, sep="", end=" ", output_stream=sys.stdout)


def label_collate(labels):
    """Collates the label inputs for the transducer prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, tf.Tensor):
        return tf.cast(labels, tf.int32)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    all_lens = [len(label) for label in labels]
    max_len = max(all_lens)

    # create a empty container with padding idx only
    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for idx, l in enumerate(labels):
        cat_labels[idx, : len(l)] = l
    labels = tf.convert_to_tensor(cat_labels, dtype=tf.int32)

    return labels
