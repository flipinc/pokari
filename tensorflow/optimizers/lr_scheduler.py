import numpy as np
import tensorflow as tf


class WarmupCosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        learning_rate,
        total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=0,
        warmup_ratio=None,
        hold_base_steps=0,
        hold_base_ratio=None,
    ):
        """Cosine decay schedule with warm up period.
        Args:
            learning_rate: base learning rate.
            total_steps: total number of training steps.
            warmup_learning_rate: initial learning rate for warm up.
            warmup_steps: number of warmup steps.
            hold_base_steps: Optional number of steps to hold base learning rate
                before decaying.

        """
        super().__init__()

        if warmup_ratio is not None and warmup_ratio > 1:
            raise ValueError("Warmup ratio must be less than 1.")

        if warmup_steps > total_steps:
            raise ValueError("Warmup steps must be less than the total steps.")

        self.learning_rate_base = learning_rate
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = (
            int(total_steps * warmup_ratio)
            if warmup_ratio is not None
            else warmup_steps
        )
        self.hold_base_steps = (
            int(total_steps * hold_base_ratio)
            if hold_base_ratio is not None
            else hold_base_steps
        )

    def __call__(self, global_step):
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    np.pi
                    * (
                        tf.cast(global_step, tf.float32)
                        - self.warmup_steps
                        - self.hold_base_steps
                    )
                    / float(self.total_steps - self.warmup_steps - self.hold_base_steps)
                )
            )
        )

        if self.hold_base_steps > 0:
            learning_rate = tf.where(
                global_step > self.warmup_steps + self.hold_base_steps,
                learning_rate,
                self.learning_rate_base,
            )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = (
                slope * tf.cast(global_step, tf.float32) + self.warmup_learning_rate
            )
            learning_rate = tf.where(
                global_step < self.warmup_steps, warmup_rate, learning_rate
            )

        return tf.where(global_step > self.total_steps, 0.0, learning_rate)
