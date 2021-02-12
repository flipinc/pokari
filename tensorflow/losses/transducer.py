import tensorflow as tf
import warprnnt_tensorflow as warprnnt


class TransducerLoss(tf.keras.losses.Loss):
    def __init__(self):
        pass

    def call(
        self,
        log_probs,
        targets,
        encoded_lens,
        decoded_lens,
    ):
        pass
