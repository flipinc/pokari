import collections

import tensorflow as tf

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


class Inference(tf.keras.layers.Layer):
    def __init__(self, text_featurizer, predictor, joint):
        super().__init__()

        self.text_featurizer = text_featurizer
        self.predictor = predictor
        self.joint = joint

    def decoder_inference(
        self, encoded_outs: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor
    ):
        """Infer function for decoder

        Args:
            encoded_outs (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence =>
                shape []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded_outs = tf.reshape(encoded_outs, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predictor.recognize(
                predicted, states
            )  # [1, 1, P], states
            ytu = tf.nn.log_softmax(
                self.joint([encoded_outs, y], training=False)
            )  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    def _perform_greedy_batch(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
        version: str = "v1",
    ):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            t_max = tf.math.reduce_max(encoded_length)

            greedy_fn = (
                self._perform_greedy if version == "v1" else self._perform_greedy_v2
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

    def _perform_greedy(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        with tf.name_scope(f"{self.name}_greedy"):
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

    def _perform_greedy_v2(
        self,
        encoded: tf.Tensor,
        encoded_length: tf.Tensor,
        predicted: tf.Tensor,
        states: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = False,
    ):
        """ Ref: https://arxiv.org/pdf/1801.00841.pdf """
        with tf.name_scope(f"{self.name}_greedy_v2"):
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
