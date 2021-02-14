from typing import Optional, Tuple

import tensorflow as tf

# TODO: instead of while_loop, rely on tensorflow's autograph
# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow
#           /python/autograph/g3doc/reference/control_flow.md#while-statements
# ref: https://www.tensorflow.org/guide/function#loops
# ref: https://github.com/tensorflow/tensorflow/issues/45337


class GreedyInference(tf.keras.layers.Layer):
    """A greedy transducer decoder.

    Batch level greedy decoding, performed auto-repressively.

    Args:
        decoder: Transducer decoder model.
        joint: Transducer joint model.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
            max_symbols_per_step: Optional int. The maximum number of symbols
            that can be added to a sequence in a single time step; if set to
            None then there is no limit.
    """

    def __init__(
        self,
        predictor,
        joint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__()

        self.predictor = predictor
        self.joint = joint

        self._blank_idx = blank_index
        self._SOS = blank_index
        self.max_symbols = max_symbols_per_step

    def _pred_step(
        self,
        label: Optional[tf.Tensor],
        state: Optional[Tuple[tf.Tensor, tf.Tensor]],
        batch_size: Optional[int] = None,
    ):
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (Optional torch.Tensor): Label or start-of-signal token. [1]
            state: (Optional torch.Tensor): RNN State vector
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, 1, H)
            hid: (h, c) where h is the final sequence state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if label is None:  # start of signal
            return self.predictor.predict(
                None, state, add_sos=False, batch_size=batch_size
            )
        else:  # label
            return self.predictor.predict(
                label, state, add_sos=False, batch_size=batch_size
            )

    def _joint_step(self, enc, pred):
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        logits = self.joint.joint(enc, pred)

        return logits

    def call(
        self,
        encoded_outs: tf.Tensor,
        encoded_lens: tf.Tensor,
        cache_rnn_state: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        mode: str = "full_context",
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoded_outs: A tensor of size (batch, features, timesteps).
            encoded_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            hypotheses: [B, T]
        """
        # Apply optional preprocessing
        encoded_outs = tf.transpose(encoded_outs, [0, 2, 1])  # (B, T, D)

        hypotheses, cache_rnn_state = self._greedy_naive_batch_decode(
            encoded_outs, encoded_lens, cache_rnn_state
        )

        return hypotheses, cache_rnn_state

    def _greedy_naive_decode(
        self, encoded_out: tf.Tensor, encoded_len: tf.Tensor, states: tf.Tensor = None
    ):
        """

        Args:
            encoded_out: [T, D]
            encoded_len: [1]

        TODO: this is ridiculously slow compared to pytorch implementation. the main
        causes are pred_step and joint_step. should do something about this

        """
        time = tf.constant(0, dtype=tf.int32)
        max_time = encoded_len

        labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
        )
        last_label = tf.fill([1, 1], self._blank_idx)

        def blank_loop_cond(
            _time, is_blank, symbols_added, _last_label, _labels, encoded_out_t, _states
        ):
            anchor = tf.constant(0, dtype=tf.int32)
            return tf.equal(is_blank, anchor) and tf.less(
                symbols_added, self.max_symbols
            )

        def blank_loop_body(
            _time, is_blank, symbols_added, _last_label, _labels, encoded_out_t, _states
        ):
            if _time == 0 and symbols_added == 0:
                decoded_out_t, next_states = self._pred_step(
                    None, _states, batch_size=1
                )
            else:
                # Perform batch step prediction of decoder, getting new states and
                # scores ("g")
                decoded_out_t, next_states = self._pred_step(
                    _last_label, _states, batch_size=1
                )

            # Batched joint step - Output = [B, V + 1]
            joint_out = self._joint_step(encoded_out_t, decoded_out_t)
            logp = joint_out[:, 0, 0, :]  # [1, V + 1]

            symbol_idx = tf.argmax(logp, axis=-1, output_type=tf.int32)  # [1]

            # TODO: should this be moved inside false_fn as recording blank_idx is
            # unnecessary
            _labels = _labels.write(_labels.size(), symbol_idx)

            def true_fn():
                is_blank = tf.constant(1, tf.int32)
                return (
                    _time,
                    is_blank,
                    symbols_added,
                    _last_label,
                    _labels,
                    encoded_out_t,
                    _states,
                )

            def false_fn():
                _states = next_states
                _last_label = tf.expand_dims(symbol_idx, axis=0)
                return (
                    _time,
                    is_blank,
                    symbols_added + 1,
                    _last_label,
                    _labels,
                    encoded_out_t,
                    _states,
                )

            return tf.cond(tf.equal(symbol_idx, self._blank_idx), true_fn, false_fn)

        def time_loop_cond(_time, _last_label, _labels, _states):
            return tf.less(_time, max_time)

        def time_loop_body(_time, _last_label, _labels, _states):
            encoded_out_t = tf.gather(encoded_out, tf.reshape(_time, shape=[1]))
            is_blank = tf.constant(0, tf.int32)
            symbols_added = tf.constant(0, tf.int32)

            _, _, _, _last_label, _labels, _, _states = tf.while_loop(
                blank_loop_cond,
                blank_loop_body,
                loop_vars=[
                    _time,
                    is_blank,
                    symbols_added,
                    _last_label,
                    _labels,
                    encoded_out_t,
                    _states,
                ],
            )

            return _time + 1, _last_label, _labels, _states

        _, _, labels, states = tf.while_loop(
            time_loop_cond,
            time_loop_body,
            loop_vars=[time, last_label, labels, states],
        )

        # [B, T, 1] -> [B, T]
        labels = tf.squeeze(labels.stack())

        return labels, states

    def _greedy_naive_batch_decode(
        self, encoded_outs: tf.Tensor, encoded_lens: tf.Tensor, states: tf.Tensor = None
    ):
        """Naive implementation of greedy batch decode, which loops over batch size.

        N_p: number of layers in predictor

        Args:
            encoded_out: [B, T, D]
            encoded_lens: [B]
            states: List[([B, D], [B, D])] * N_p

        """
        batch_idx = tf.constant(0, dtype=tf.int32)
        batch_size = tf.shape(encoded_outs)[0]

        if states is None:
            states = self.predictor.initialize_state(batch_size, training=False)

        def batch_loop_cond(_batch_idx, _labels, _states):
            return tf.less(_batch_idx, batch_size)

        labels = tf.TensorArray(size=batch_size, dtype=tf.int32)
        new_states = tf.TensorArray(size=batch_size, dtype=encoded_outs.dtype)

        def batch_loop_body(_batch_idx, _labels, _new_states):
            _labels_one, _states_one = self._greedy_naive_decode(
                encoded_outs[_batch_idx],
                encoded_lens[_batch_idx],
                states=tf.gather(states, [batch_idx], axis=2),
            )

            _labels = _labels.write(_batch_idx, _labels_one)
            _new_states = _new_states.write(_batch_idx, _states_one)

            return _batch_idx + 1, _labels, _new_states

        _, _labels, new_states = tf.while_loop(
            batch_loop_cond,
            batch_loop_body,
            loop_vars=[batch_idx, labels, new_states],
        )

        return labels.stack(), new_states.stack()

    def _greedy_batch_decode(encoded_out: tf.Tensor, encoded_lens: tf.Tensor):
        """Greedy batch decoding in parallel"""

        # TODO: use greedy naive decode as base

        pass
