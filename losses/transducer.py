import torch
import torch.nn as nn


# temporary class
class warprnnt(nn.Module):
    @staticmethod
    def RNNTLoss(self, blank, reduction):
        return 1


class RNNTLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, reduction="mean_batch"):
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
            num_classes: Number of target classes for the joint network to predict.
                (Excluding the RNN-T blank token).

            reduction: Type of reduction to perform on loss. Possibly values are `mean`,
                `sum` or None. None will return a torch vector comprising the individual
                loss values of the batch.
        """
        super(RNNTLoss, self).__init__()

        if reduction not in [None, "mean", "sum", "mean_batch"]:
            raise ValueError("`reduction` must be one of [mean, sum, mean_batch]")

        self._blank = num_classes
        self.reduction = reduction
        self._loss = warprnnt.RNNTLoss(blank=self._blank, reduction="none")

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Cast to int 32
        targets = targets.int()
        input_lengths = input_lengths.int()
        target_lengths = target_lengths.int()

        max_logit_len = input_lengths.max()
        max_targets_len = target_lengths.max()

        # Force cast joint to float32
        if log_probs.dtype != torch.float32:
            log_probs = log_probs.float()

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        # of the log_probs tensor, therefore we increment the input_lengths by the
        # difference. This difference is generally small.
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
        loss = self._loss(
            acts=log_probs,
            labels=targets,
            act_lens=input_lengths,
            label_lens=target_lengths,
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
            input_lengths,
            target_lengths,
        )

        return loss
