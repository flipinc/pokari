import collections

import tensorflow as tf

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


class Inference(tf.keras.layers.Layer):
    def __init__(self, text_featurizer, predictor, joint, max_symbols: int = 1):
        super().__init__()

        self.text_featurizer = text_featurizer

        self.predictor = predictor
        self.joint = joint

        self.max_symbols = max_symbols

    def decoder_inference(
        self, encoded_outs: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor
    ):
        """Infer function for decoder

        Args:
            encoded_outs (tf.Tensor): output of encoder at each time step [D]
            predicted (tf.Tensor): last character index of predicted sequence []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded_outs = tf.reshape(encoded_outs, [1, 1, -1])  # [D] => [1, 1, D]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predictor.recognize(
                predicted, states
            )  # [1, 1, P], states
            ytu = tf.nn.log_softmax(
                self.joint([encoded_outs, y], training=False)
            )  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    def decoder_batch_inference(self, encoded_outs, predicted, states):
        """

        Args:
            encoded_outs: [B, 1, D]
            predicted: [B, 1]

        """
        # [B, 1, D_p]
        y, states = self.predictor.recognize(predicted, states)
        # [B, 1, 1, V]
        ytu = tf.nn.log_softmax(self.joint([encoded_outs, y], training=False))
        # [B, V]
        ytu = ytu[:, 0, 0, :]

        return ytu, states

    def _greedy_batch_decode(
        self, encoded_outs: tf.Tensor, encoded_lens: tf.Tensor, states: tf.Tensor = None
    ):
        """Greedy batch decoding in parallel

        This is almost 2~4 times as fast as greedy_naive_batch_decode.

        TODO: This impl is not completed yet. `labels` include unnecessary symbols.

        Args:
            encoded_outs: [B, T, D_e]
            encoded_lens: [B]
            states: [N, 2, B, D_p]

        """
        bs = tf.shape(encoded_outs)[0]
        t_max = tf.shape(encoded_outs)[1]
        labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
        )

        last_label = tf.fill([bs, 1], self.text_featurizer.blank)

        blank_indices = tf.fill([bs], 0)
        blank_mask = tf.fill([bs], 0)
        one = tf.constant(1, tf.int32)
        zero = tf.constant(0, tf.int32)
        states = (
            states
            if states is not None
            else self.predictor.get_initial_state(bs, False)
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
                # [B, V]
                ytu, next_states = self.decoder_batch_inference(
                    encoded_outs=encoded_out_t,
                    predicted=last_label,
                    states=states,
                )

                symbols = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # [B]

                symbol_is_blank = tf.cast(
                    symbols == self.text_featurizer.blank, tf.int32
                )
                # bitwise_or return [1, B] so get rid of first dim -> [B]
                blank_mask = tf.squeeze(
                    tf.bitwise.bitwise_or(blank_mask, symbol_is_blank)
                )

                # labels = labels.write(labels.size(), symbols)

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

                    # recover prior state for all samples which predicted blank now/past
                    # if states is not None:
                    unchanged_states = tf.gather(states, blank_indices, axis=2)
                    changes_states = tf.gather(next_states, non_blank_indices, axis=2)

                    unordered_next_states = tf.concat(
                        [unchanged_states, changes_states], axis=2
                    )
                    next_states = tf.gather(
                        unordered_next_states, ordered_indicies, axis=2
                    )

                    # recover prior predicted label for all samples which predicted
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

                    # update new label and hidden state for next iteration
                    last_label = tf.expand_dims(next_symbols, axis=-1)
                    states = next_states

                    # update label
                    counter = tf.constant(0, tf.int32)
                    new_label_t = tf.TensorArray(
                        tf.int32,
                        size=bs,
                        clear_after_read=True,
                    )
                    # prev_label_t = labels.read(t)
                    for symbol in next_symbols:
                        # if not blank add to label
                        if tf.equal(blank_mask[counter], zero):
                            new_label_t = new_label_t.write(counter, symbol)
                        else:
                            new_label_t = new_label_t.write(
                                counter, self.text_featurizer.blank
                            )
                        # else:
                        #     # already finished predicting all timesteps
                        #     if tf.math.greater_equal(t, encoded_lens[counter]):
                        #         new_label_t = new_label_t.write(
                        #             counter, self.text_featurizer.blank
                        #         )
                        #     # if not finished, keep the previous label
                        #     else:
                        #         new_label_t = new_label_t.write(
                        #             counter, prev_label_t[counter]
                        #         )
                        counter += 1
                    new_label_t = new_label_t.stack()
                    labels = labels.write(labels.size(), new_label_t)

                    symbols_added += 1

        # [B, T]
        labels = tf.transpose(labels.stack(), (1, 0))
        labels = self.text_featurizer.iextract(labels)

        # return labels, states
        return labels

    def _greedy_naive_batch_decode(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
        version: str = "v1",
    ):
        """

        Args:
            encoded: [B, T, D]

        """
        with tf.name_scope(f"{self.name}_greedy_naive_batch_decode"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            t_max = tf.math.reduce_max(encoded_length)

            greedy_fn = (
                self._greedy_decode if version == "v1" else self._greedy_decode_v2
            )

            decoded = tf.TensorArray(
                dtype=tf.int32,
                size=total_batch,
                dynamic_size=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([None]),
            )

            def condition(batch, _):
                return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = greedy_fn(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    states=self.predictor.get_initial_state(batch_size=1),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                prediction = tf.pad(
                    hypothesis.prediction,
                    paddings=[[0, t_max - tf.shape(hypothesis.prediction)[0]]],
                    mode="CONSTANT",
                    constant_values=self.text_featurizer.blank,
                )
                decoded = decoded.write(batch, prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            return self.text_featurizer.iextract(decoded.stack())

    def _greedy_decode(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        with tf.name_scope(f"{self.name}_greedy_decode"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=total,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded_outs=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                # something is wrong with tflite that drop support for tf.cond
                # def equal_blank_fn(): return _hypothesis.index, _hypothesis.states
                # def non_equal_blank_fn(): return _predict, _states  # update if the
                # new prediction is a non-blank
                # _index, _states = tf.cond(tf.equal(_predict, blank), equal_blank_fn,
                # non_equal_blank_fn)

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(
                    index=_index, prediction=_prediction, states=_states
                )

                return _time + 1, _hypothesis

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )

    def _greedy_decode_v2(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        """ Ref: https://arxiv.org/pdf/1801.00841.pdf """
        with tf.name_scope(f"{self.name}_greedy_decode_v2"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)
                _time = tf.where(_equal, _time + 1, _time)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(
                    index=_index, prediction=_prediction, states=_states
                )

                return _time, _hypothesis

            time, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=[time, hypothesis],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
            )

            return Hypothesis(
                index=hypothesis.index,
                prediction=hypothesis.prediction.stack(),
                states=hypothesis.states,
            )
