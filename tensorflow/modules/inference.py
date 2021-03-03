import collections

import tensorflow as tf

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


class Inference:
    def __init__(self, text_featurizer, predictor, joint, max_symbols: int = 1):
        super().__init__()

        self.text_featurizer = text_featurizer

        self.predictor = predictor
        self.joint = joint

        self.max_symbols = max_symbols

    def decoder_inference(
        self, encoded_out: tf.Tensor, prev_token: tf.Tensor, cache_states: tf.Tensor
    ):
        """Infer function for decoder

        Args:
            encoded_outs (tf.Tensor): output of encoder at each time step [D]
            prev_token (tf.Tensor): last character index of predicted sequence []
            cache_states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope("inference_decoder"):
            encoded_outs = tf.reshape(encoded_out, [1, 1, -1])  # [D] => [1, 1, D]
            prev_token = tf.reshape(prev_token, [1, 1])  # [] => [1, 1]
            y, cache_states = self.predictor.stream(
                prev_token, cache_states
            )  # [1, 1, P], states
            ytu = tf.nn.log_softmax(
                self.joint([encoded_outs, y], training=False)
            )  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, cache_states

    def decoder_batch_inference(self, encoded_outs, prev_tokens, cache_states):
        """

        Args:
            encoded_outs: [B, 1, D]
            prev_tokens: [B, 1]

        """
        with tf.name_scope("inference_batch_decoder"):
            # [B, 1, D_p]
            predictor_outs, cache_states = self.predictor.stream(
                prev_tokens, cache_states
            )
            # [B, T, U, V]
            joint_outs = self.joint([encoded_outs, predictor_outs], training=False)
            # [B, 1, 1, V]
            ytu = tf.nn.log_softmax(joint_outs)
            # [B, V]
            ytu = ytu[:, 0, 0, :]

            return ytu, cache_states

    def greedy_batch_decode(
        self,
        encoded_outs: tf.Tensor,
        encoded_lens: tf.Tensor,
        prev_tokens: tf.Tensor = None,
        cache_states: tf.Tensor = None,
    ):
        """Greedy batch decoding in parallel

        This is almost 2~4 times as fast as greedy_naive_batch_decode.

        TODO: This impl is not completed yet. `labels` include unnecessary symbols.

        Args:
            encoded_outs: [B, T, D_e]
            encoded_lens: [B]
            prev_tokens: [B, 1]
            cache_states: [N, 2, B, D_p]

        """
        bs = tf.shape(encoded_outs)[0]
        t_max = tf.shape(encoded_outs)[1]
        labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            clear_after_read=True,
        )

        # after some training iterations (usually very beginning of the
        # first epoch), model starts to predict all blanks. So, a blank has to be
        # inside labels in order to stack(). All blanks will be removed afterwards.
        labels = labels.write(0, tf.fill([bs], 0))

        last_label = (
            prev_tokens
            if prev_tokens is not None
            else tf.fill([bs, 1], self.text_featurizer.blank)
        )

        blank_mask = tf.fill([bs], 0)
        zero = tf.constant(0, tf.int32)
        states = (
            cache_states
            if cache_states is not None
            else self.predictor.get_initial_state(bs, False)
        )

        anchor = tf.constant(0, dtype=tf.int32)
        for t in tf.range(t_max):
            is_blank = tf.constant(0, tf.int32)
            symbols_added = tf.constant(0, tf.int32)

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
                    prev_tokens=last_label,
                    cache_states=states,
                )

                symbols = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # [B]

                symbol_is_blank = tf.cast(
                    symbols == self.text_featurizer.blank, tf.int32
                )

                # [B]
                blank_mask = tf.bitwise.bitwise_or(blank_mask, symbol_is_blank)

                if tf.reduce_all(tf.cast(blank_mask, tf.bool)):
                    is_blank = tf.constant(1, tf.int32)
                else:
                    # recover prior state for all samples which predicted blank now/past
                    states_blank_mask = tf.cast(
                        tf.reshape(blank_mask, [1, 1, bs, 1]), tf.float32
                    )
                    unchanged_states = states * states_blank_mask
                    inverse_states_blank_mask = states_blank_mask * -1.0 + 1.0
                    changed_states = next_states * inverse_states_blank_mask
                    next_states = unchanged_states + changed_states

                    # recover prior predicted label for all samples which predicted
                    symbol_blank_mask = blank_mask
                    unchanged_symbols = tf.squeeze(last_label, axis=-1) * blank_mask
                    inverse_symbol_blank_mask = symbol_blank_mask * -1 + 1
                    changed_symbols = symbols * inverse_symbol_blank_mask
                    next_symbols = unchanged_symbols + changed_symbols

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
                    for symbol in next_symbols:
                        # if not blank add to label
                        if tf.equal(blank_mask[counter], zero):
                            new_label_t = new_label_t.write(counter, symbol)
                        else:
                            new_label_t = new_label_t.write(
                                counter, self.text_featurizer.blank
                            )
                        counter += 1
                    new_label_t = new_label_t.stack()
                    labels = labels.write(labels.size(), new_label_t)

                    symbols_added += 1

        # [T, B]
        labels = labels.stack()
        # [B, T]
        labels = tf.transpose(labels, (1, 0))
        labels = self.text_featurizer.iextract(labels)

        return labels, last_label, states

    def greedy_naive_batch_decode(
        self,
        encoded_outs: tf.Tensor,
        encoded_lens: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
        version: str = "v1",
    ):
        """

        Args:
            encoded_outs: [B, T, D]

        """
        with tf.name_scope("greedy_naive_batch_decode"):
            total_batch = tf.shape(encoded_outs)[0]
            batch = tf.constant(0, dtype=tf.int32)

            t_max = tf.math.reduce_max(encoded_lens)

            greedy_fn = self.greedy_decode if version == "v1" else self.greedy_decode_v2

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
                    encoded_out=encoded_outs[batch],
                    encoded_len=encoded_lens[batch],
                    prev_token=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    cache_states=self.predictor.get_initial_state(batch_size=1),
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

    def greedy_decode(
        self,
        encoded_out: tf.Tensor,
        encoded_len: tf.Tensor,
        prev_token: tf.Tensor,
        cache_states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        with tf.name_scope("greedy_decode"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_len

            hypothesis = Hypothesis(
                index=prev_token,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=total,
                    dynamic_size=False,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=cache_states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    # ref: https://github.com/TensorSpeech/TensorFlowASR/issues/17
                    encoded_out=tf.gather_nd(encoded_out, tf.reshape(_time, shape=[1])),
                    prev_token=_hypothesis.index,
                    cache_states=_hypothesis.states,
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

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

    def greedy_decode_v2(
        self,
        encoded_out: tf.Tensor,
        encoded_len: tf.Tensor,
        prev_token: tf.Tensor,
        cache_states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        """ Ref: https://arxiv.org/pdf/1801.00841.pdf """
        with tf.name_scope("greedy_decode_v2"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_len

            hypothesis = Hypothesis(
                index=prev_token,
                prediction=tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    clear_after_read=False,
                    element_shape=tf.TensorShape([]),
                ),
                states=cache_states,
            )

            def condition(_time, _hypothesis):
                return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded_out=tf.gather_nd(encoded_out, tf.reshape(_time, shape=[1])),
                    prev_token=_hypothesis.index,
                    cache_states=_hypothesis.states,
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
