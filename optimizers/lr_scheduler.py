import logging
import math
import warnings

import torch


class WarmupPolicy(torch.optim.lr_scheduler._LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert (
            warmup_ratio is None or max_steps is not None
        ), "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler,"
                "please use `get_last_lr()`.",
                UserWarning,
            )

        step = self.last_epoch

        if step <= self.warmup_steps:
            lr_val = (step + 1) / (self.warmup_steps + 1)
            return [initial_lr * lr_val for initial_lr in self.base_lrs]

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class CosineAnnealing(WarmupPolicy):
    def __init__(self, optimizer, *, max_steps, min_lr=0, last_epoch=-1, **kwargs):
        super().__init__(
            optimizer=optimizer,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
            **kwargs,
        )

    def _get_lr(self, step):
        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the"
                    "minimum learning rate."
                )

        def _cosine_annealing(initial_lr, step, max_steps, min_lr):
            mult = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            out_lr = (initial_lr - min_lr) * mult + min_lr
            return out_lr

        new_lrs = [
            _cosine_annealing(
                initial_lr=initial_lr,
                step=step - self.warmup_steps,
                max_steps=self.max_steps - self.warmup_steps,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


def compute_max_steps(
    max_epochs,
    accumulate_grad_batches,
    limit_train_batches,
    num_workers,
    num_samples,
    batch_size,
    drop_last,
):
    _round = math.floor if drop_last else math.ceil

    sampler_num_samples = math.ceil(num_samples / num_workers)

    if drop_last and num_workers > 1:
        logging.warning(
            "Please note that drop_last is broken in pytorch 1.6.0. "
            "We will fix when pytorch 1.7.0 is released"
        )
        # TODO: Master verion, not in pytorch 1.6.0
        # sampler_num_samples = math.ceil((num_samples - num_workers)/ num_workers)

    steps_per_epoch = _round(sampler_num_samples / batch_size)
    if isinstance(limit_train_batches, int) or limit_train_batches == 0.0:
        steps_per_epoch = min(steps_per_epoch, int(limit_train_batches))
    elif steps_per_epoch != float("inf"):
        # limit_train_batches is a percentage of batches per epoch
        steps_per_epoch = int(steps_per_epoch * limit_train_batches)
        if accumulate_grad_batches == 1:
            steps_per_epoch = max(steps_per_epoch, 1)

    return math.ceil(steps_per_epoch / accumulate_grad_batches) * max_epochs
