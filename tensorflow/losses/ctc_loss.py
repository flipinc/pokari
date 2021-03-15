import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class CTCLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank,
        global_batch_size,
        reduction=losses_utils.ReductionV2.NONE,
        name=None,
    ):
        super().__init__(reduction=reduction, name=name)
        self.blank = blank
        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        logits = y_pred["logits"]
        logit_lens = y_pred["logit_lens"]
        labels = y_true["labels"]
        label_lens = y_true["label_lens"]
        loss = tf.nn.ctc_loss(
            labels=tf.cast(labels, tf.int32),
            logit_length=tf.cast(logit_lens, tf.int32),
            logits=tf.cast(logits, tf.float32),
            label_length=tf.cast(label_lens, tf.int32),
            logits_time_major=False,
            blank_index=self.blank,
        )
        return tf.nn.compute_average_loss(
            loss, global_batch_size=self.global_batch_size
        )
