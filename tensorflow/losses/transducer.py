import tensorflow as tf
from warprnnt_tensorflow import rnnt_loss as warp_rnnt_loss


class TransducerLoss(tf.keras.losses.Loss):
    def __init__(self, vocab_size, reduction="mean_batch"):
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
        super().__init__()

        if reduction not in [None, "mean", "sum", "mean_batch"]:
            raise ValueError("`reduction` must be one of [mean, sum, mean_batch]")

        self._blank = vocab_size
        self.reduction = reduction

    def forward(self, log_probs, targets, encoded_lens, decoded_lens):
        # Cast to int 32
        targets = tf.cast(targets, tf.int32)
        encoded_lens = tf.convert_to_tensor(encoded_lens, tf.int32)
        decoded_lens = tf.convert_to_tensor(decoded_lens, tf.int32)

        max_logit_len = encoded_lens.max()
        max_targets_len = decoded_lens.max()

        # Force cast joint to float32
        if log_probs.dtype != tf.float32:
            log_probs = tf.cast(log_probs, tf.float32)

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        # of the log_probs tensor, therefore we increment the encoded_lens by the
        # difference. This difference is generally small.
        print(log_probs.shape[1], max_logit_len)
        if log_probs.shape[1] != max_logit_len:
            log_probs = log_probs.narrow(
                dim=1, start=0, length=max_logit_len
            ).contiguous()

        # Reduce transcript length to correct alignment if additional padding was
        # applied.
        # Transcript: [B, L] -> [B, L']; If L' < L
        if targets.shape[1] != max_targets_len:
            targets = targets.narrow(dim=1, start=0, length=max_targets_len)

        # Loss reduction can be dynamic, so set it prior to call
        if self.reduction != "mean_batch":
            self._loss.reduction = self.reduction

        # Compute RNNT loss
        loss = warp_rnnt_loss(
            acts=log_probs,
            labels=targets,
            act_lens=encoded_lens,
            label_lens=decoded_lens,
            blank_label=self._blank,
        )

        # Loss reduction can be dynamic, so reset it after call
        if self.reduction != "mean_batch":
            self._loss.reduction = "none"

        # Loss reduction only for mean_batch mode
        if self.reduction == "mean_batch":
            loss = torch.mean(loss)

        # del new variables that may have been created
        del (
            log_probs,
            targets,
            encoded_lens,
            decoded_lens,
        )

        return loss
