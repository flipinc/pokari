import tensorflow as tf
from warprnnt_tensorflow import rnnt_loss as warp_rnnt_loss


class TransducerLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size, vocab_size):
        """
        RNN-T Loss function based on https://github.com/HawkAaron/warp-transducer.

        Note:
            Requires the pytorch bindings to be installed prior to calling this class.

        Warning:
            In the case that GPU memory is exhausted in order to compute RNNTLoss,
            it might cause a core dump at the cuda level with the following
            error message.

            ```
                ...
                costs = costs.to(acts.device)
            RuntimeError: CUDA error: an illegal memory access was encountered
            terminate called after throwing an instance of 'c10::Error'
            ```

            Please kill all remaining python processes after this point, and use a
            smaller batch size for train, validation and test sets so that CUDA memory
            is not exhausted.

        Args:
            vocab_size: Number of target classes for the joint network to predict.
                (Excluding the RNN-T blank token).

            reduction: Type of reduction to perform on loss. Possibly values are `mean`,
                `sum` or None. None will return a torch vector comprising the individual
                loss values of the batch.
        """
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

        self.batch_size = batch_size
        self.blank = vocab_size

    def call(self, y_true, y_pred):
        log_probs, encoded_lens = y_true
        targets, decoded_lens = y_pred

        # TODO(keisuke): if length mismatches between log_probs <-> decoded_lens or
        # targets <-> encoded_lens, align shapes just like in pytorch implementation

        # Cast to int 32
        if targets.dtype != tf.float32:
            targets = tf.cast(targets, tf.int32)
        if encoded_lens.dtype != tf.float32:
            encoded_lens = tf.cast(encoded_lens, tf.int32)
        if decoded_lens.dtype != tf.float32:
            decoded_lens = tf.cast(decoded_lens, tf.int32)

        # Force cast joint to float32
        if log_probs.dtype != tf.float32:
            log_probs = tf.cast(log_probs, tf.float32)

        # Compute RNNT loss
        loss = warp_rnnt_loss(
            acts=log_probs,
            labels=targets,
            input_lengths=encoded_lens,
            label_lengths=decoded_lens,
            blank_label=self.blank,
        )

        loss = tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)

        return loss
