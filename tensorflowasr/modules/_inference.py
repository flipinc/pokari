import collections
from typing import Optional, Tuple

import tensorflow as tf

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


class Inference(tf.keras.layers.Layer):
    def __init__(
        self,
        predictor,
        joint,
        blank_index: int = 0,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__()

        self.predictor = predictor
        self.joint = joint

        self.blank_idx = blank_index
        self.max_symbols = max_symbols_per_step

    #######
    # New impl
    #######

    def _greedy_naive_decode(
        self,
        encoded_out: tf.Tensor,
        encoded_length: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        """Naive implementation of greedy decoding. Only Accepts Batch_size = 1

        For better readability, we rely on Autograph as much as possible. See Github's
        offical documentation for more info on Autograph.

        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph
        /g3doc/reference

        Args:
            encoded_out: [T, D]
            encoded_len: [1]

        TODO: this is ridiculously slow compared to pytorch implementation. the main
        causes are pred_step and joint_step. should do something about this

        """
        hypothesis = Hypothesis(
            index=predicted,
            prediction=tf.TensorArray(
                dtype=tf.int32,
                size=encoded_length,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([]),
            ),
            states=states,
        )

        time = tf.constant(0, dtype=tf.int32)
        anchor = tf.constant(0, dtype=tf.int32)
        for encoded_out_t in encoded_out:
            is_blank = tf.constant(0, tf.int32)
            symbols_added = tf.constant(0, tf.int32)

            while tf.equal(is_blank, anchor) and tf.less(
                symbols_added, self.max_symbols
            ):
                ytu, states = self.decoder_inference(
                    encoded_outs=encoded_out_t,
                    predicted=hypothesis.index,
                    states=hypothesis.states,
                )

                symbol_idx = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # [1]

                if tf.equal(symbol_idx, self.text_featurizer.blank):
                    index = hypothesis.index
                    states = hypothesis.states
                    is_blank = tf.constant(1, tf.int32)
                else:
                    index = symbol_idx
                    symbols_added += 1

                prediction = hypothesis.prediction.write(time, symbol_idx)
                hypothesis = Hypothesis(
                    index=index, prediction=prediction, states=states
                )

            time += 1

        return Hypothesis(
            index=hypothesis.index,
            prediction=hypothesis.prediction.stack(),
            states=hypothesis.states,
        )

    def _greedy_naive_batch_decode(
        self,
        encoded_outs: tf.Tensor,
        encoded_lens: tf.Tensor,
        states: tf.Tensor = None,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
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
        t_max = tf.shape(encoded_outs)[1]

        if states is None:
            states = self.predictor.get_initial_state(batch_size, training=False)

        decoded = tf.TensorArray(
            dtype=tf.int32,
            size=batch_size,
            dynamic_size=False,
            clear_after_read=False,
            element_shape=tf.TensorShape([None]),
        )

        for batch_idx in tf.range(batch_size):
            hypothesis = self._greedy_naive_decode(
                encoded_outs[batch_idx],
                encoded_lens[batch_idx],
                predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                states=tf.gather(states, [batch_idx], axis=2),
            )

            # TODO: do something better
            # each batch may have different label length so pad to maximum length
            prediction = tf.pad(
                hypothesis.prediction,
                paddings=[
                    [0, self.max_symbols * t_max - tf.shape(hypothesis.prediction)[0]]
                ],
                mode="CONSTANT",
                constant_values=self.text_featurizer.blank,
            )

            decoded = decoded.write(batch_idx, prediction)

        return self.text_featurizer.iextract(decoded.stack())

    #######
    # Old impl
    #######

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

        hypotheses, cache_rnn_state = self._greedy_batch_decode(
            encoded_outs, encoded_lens, cache_rnn_state
        )

        return hypotheses, cache_rnn_state

    def __greedy_naive_decode(
        self, encoded_out: tf.Tensor, encoded_len: tf.Tensor, states: tf.Tensor = None
    ):
        """Naive implementation of greedy decoding. Only Accepts Batch_size = 1

        For better readability, we rely on Autograph as much as possible. See Github's
        offical documentation for more info on Autograph.

        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph
        /g3doc/reference

        Args:
            encoded_out: [T, D]
            encoded_len: [1]

        TODO: this is ridiculously slow compared to pytorch implementation. the main
        causes are pred_step and joint_step. should do something about this

        """
        labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
        )
        last_label = tf.fill([1, 1], self.blank_idx)

        time = tf.constant(0, dtype=tf.int32)
        anchor = tf.constant(0, dtype=tf.int32)
        for encoded_out_t in encoded_out:
            is_blank = tf.constant(0, tf.int32)
            symbols_added = tf.constant(0, tf.int32)

            while tf.equal(is_blank, anchor) and tf.less(
                symbols_added, self.max_symbols
            ):
                if tf.equal(time, 0) and tf.equal(symbols_added, 0):
                    decoded_out_t, next_states = self._pred_step(
                        None, states, batch_size=1
                    )
                else:
                    decoded_out_t, next_states = self._pred_step(
                        last_label, states, batch_size=1
                    )

                # [1, 1, 1, V + 1]
                joint_out = self._joint_step(
                    tf.expand_dims(encoded_out_t, axis=0), decoded_out_t
                )
                # [1, V + 1]
                logp = joint_out[:, 0, 0, :]

                symbol_idx = tf.argmax(logp, axis=-1, output_type=tf.int32)  # [1]

                labels = labels.write(labels.size(), symbol_idx)

                if tf.equal(symbol_idx, self.blank_idx):
                    is_blank = tf.constant(1, tf.int32)
                else:
                    states = next_states
                    last_label = tf.expand_dims(symbol_idx, axis=0)
                    symbols_added += 1

        # [B, T, 1] -> [B, T]
        labels = tf.squeeze(labels.stack())

        return labels, states

    def __greedy_naive_batch_decode(
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
        t_max = tf.shape(encoded_outs)[1]

        if states is None:
            states = self.predictor.initialize_state(batch_size, training=False)

        labels = tf.TensorArray(size=batch_size, dtype=tf.int32)
        new_states = tf.TensorArray(size=batch_size, dtype=encoded_outs.dtype)

        for batch_idx in tf.range(batch_size):
            _labels_one, _states_one = self._greedy_naive_decode(
                encoded_outs[batch_idx],
                encoded_lens[batch_idx],
                states=tf.gather(states, [batch_idx], axis=2),
            )

            # TODO: do something better
            # each batch may have different label length so pad to maximum length
            _labels_one = tf.pad(
                _labels_one,
                paddings=[[0, self.max_symbols * t_max - tf.shape(_labels_one)[0]]],
                mode="CONSTANT",
                constant_values=self.blank_idx,
            )

            labels = labels.write(batch_idx, _labels_one)
            new_states = new_states.write(batch_idx, _states_one)

        return labels.stack(), new_states.stack()

    def __greedy_batch_decode(
        self, encoded_outs: tf.Tensor, encoded_lens: tf.Tensor, states: tf.Tensor = None
    ):
        """Greedy batch decoding in parallel"""
        bs = tf.shape(encoded_outs)[0]
        t_max = tf.shape(encoded_outs)[1]
        labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
        )

        last_label = tf.fill([bs, 1], self.blank_idx)

        blank_indices = tf.fill([bs], 0)
        blank_mask = tf.fill([bs], 0)
        one = tf.constant(1, tf.int32)
        states = (
            states if states is not None else self.predictor.initialize_state(bs, False)
        )

        anchor = tf.constant(0, dtype=tf.int32)
        for t in tf.range(t_max):
            is_blank = tf.constant(0, tf.int32)
            symbols_added = tf.constant(0, tf.int32)

            # mask buffers
            blank_mask = tf.fill(tf.shape(blank_mask), 0)

            # Forcibly mask with "blank" tokens, for all sample where
            # current time step T > seq_len
            blank_mask = tf.cast(t >= encoded_lens, tf.int32)

            # get encoded_outs at timestep t -> [B, 1, D]
            encoded_out_t = tf.gather(encoded_outs, [t], axis=1)

            while tf.equal(is_blank, anchor) and tf.less(
                symbols_added, self.max_symbols
            ):
                if tf.equal(t, 0) and tf.equal(symbols_added, 0):
                    decoded_out_t, next_states = self._pred_step(
                        None, states, batch_size=bs
                    )
                else:
                    decoded_out_t, next_states = self._pred_step(
                        last_label, states, batch_size=bs
                    )

                # [B, 1, 1, V + 1]
                joint_out = self._joint_step(encoded_out_t, decoded_out_t)
                # [B, V + 1]
                logp = joint_out[:, 0, 0, :]

                symbols = tf.argmax(logp, axis=-1, output_type=tf.int32)  # [B]

                symbol_is_blank = tf.cast(symbols == self.blank_idx, tf.int32)
                # bitwise_or return [1, B] so get rid of first dim -> [B]
                blank_mask = tf.squeeze(
                    tf.bitwise.bitwise_or(blank_mask, symbol_is_blank)
                )

                labels = labels.write(labels.size(), symbols)

                if tf.reduce_all(tf.cast(blank_mask, tf.bool)):
                    is_blank = tf.constant(1, tf.int32)
                else:
                    blank_indices = tf.squeeze(
                        tf.where(tf.cast(tf.equal(blank_mask, one), tf.int32)),
                        axis=-1,
                    )
                    non_blank_indices = tf.squeeze(
                        tf.where(tf.cast(tf.not_equal(blank_mask, one), tf.int32)),
                        axis=-1,
                    )
                    ordered_indicies = tf.concat(
                        [blank_indices, non_blank_indices], axis=0
                    )

                    # Recover prior state for all samples which predicted blank now/past
                    # if states is not None:
                    unchanged_states = tf.gather(states, blank_indices, axis=2)
                    changes_states = tf.gather(next_states, non_blank_indices, axis=2)

                    unordered_next_states = tf.concat(
                        [unchanged_states, changes_states], axis=2
                    )
                    next_states = tf.gather(
                        unordered_next_states, ordered_indicies, axis=2
                    )

                    # Recover prior predicted label for all samples which predicted
                    # blank now/past
                    unchanged_symbols = tf.gather(last_label, blank_indices)
                    # [?, 1] -> [?], where ? is the number of blank_indices
                    unchanged_symbols = tf.squeeze(unchanged_symbols, axis=-1)
                    changed_symbols = tf.gather(symbols, non_blank_indices)

                    unordered_next_symbols = tf.concat(
                        [unchanged_symbols, changed_symbols], axis=0
                    )
                    next_symbols = tf.gather(
                        unordered_next_symbols, ordered_indicies, axis=0
                    )

                    # Update new label and hidden state for next iteration
                    last_label = tf.expand_dims(next_symbols, axis=-1)
                    states = next_states

                    symbols_added += 1

        # [B, T, 1] -> [B, T]
        labels = tf.squeeze(labels.stack())

        return labels, states
