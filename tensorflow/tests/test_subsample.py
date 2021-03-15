import tensorflow as tf
from modules.subsample import StackSubsample


class TestSubsample:
    def test_stack_subsample(self):
        stack_subsample = StackSubsample(subsampling_factor=4)

        tf_before = tf.random.normal((1, 120, 80))  # [B, T, D]
        tf_after = stack_subsample(tf_before)

        before = tf_before[0, 0:4, :]
        after = tf_after[0, 0, :]

        assert tf.reduce_sum(before) == tf.reduce_sum(after)
