import collections

import tensorflow as tf
from tensorflow.keras import mixed_precision as mxp

from joint import TransducerJoint
from loss import RnntLoss
from predictor import TransducerPrediction
from subsampling import TimeReduction
from utils import (get_reduced_length, get_rnn, merge_two_last_dims,
                   pad_prediction_tfarray)

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


class StreamingTransducer(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        encoder_reductions: dict = {0: 3, 1: 2},
        encoder_dmodel: int = 640,
        encoder_nlayers: int = 8,
        encoder_rnn_type: str = "lstm",
        encoder_rnn_units: int = 2048,
        encoder_layer_norm: bool = True,
        prediction_embed_dim: int = 320,
        prediction_embed_dropout: float = 0,
        prediction_num_rnns: int = 2,
        prediction_rnn_units: int = 2048,
        prediction_rnn_type: str = "lstm",
        prediction_layer_norm: bool = True,
        prediction_projection_units: int = 640,
        joint_dim: int = 640,
        joint_activation: str = "tanh",
        prejoint_linear: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="StreamingTransducer",
        **kwargs,
    ):
        super().__init__()

        self.encoder = StreamingTransducerEncoder(
            reductions=encoder_reductions,
            dmodel=encoder_dmodel,
            nlayers=encoder_nlayers,
            rnn_type=encoder_rnn_type,
            rnn_units=encoder_rnn_units,
            layer_norm=encoder_layer_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder",
        )
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=prediction_embed_dim,
            embed_dropout=prediction_embed_dropout,
            num_rnns=prediction_num_rnns,
            rnn_units=prediction_rnn_units,
            rnn_type=prediction_rnn_type,
            rnn_implementation=prediction_rnn_type,
            layer_norm=prediction_layer_norm,
            projection_units=prediction_projection_units,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_prediction",
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            activation=joint_activation,
            prejoint_linear=prejoint_linear,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_joint",
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor

    def _build(self, input_shape):
        features = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        input_length = tf.keras.Input(shape=[], dtype=tf.int32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        pred_length = tf.keras.Input(shape=[], dtype=tf.int32)
        self(
            {
                "input": features,
                "input_length": input_length,
                "prediction": pred,
                "prediction_length": pred_length,
            },
            training=True,
        )

    def call(self, inputs, training=False, **kwargs):
        features = inputs["input"]
        prediction = inputs["prediction"]
        prediction_length = inputs["prediction_length"]
        enc = self.encoder(features, training=training, **kwargs)
        pred = self.predict_net(
            [prediction, prediction_length], training=training, **kwargs
        )
        outputs = self.joint_net([enc, pred], training=training, **kwargs)
        return {
            "logit": outputs,
            "logit_length": get_reduced_length(
                inputs["input_length"], self.time_reduction_factor
            ),
        }

    def compile(
        self,
        optimizer,
        global_batch_size,
        blank=0,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        **kwargs,
    ):
        loss = RnntLoss(blank=blank, global_batch_size=global_batch_size)
        optimizer_with_scale = mxp.experimental.LossScaleOptimizer(
            tf.keras.optimizers.get(optimizer), "dynamic"
        )
        super(StreamingTransducer, self).compile(
            optimizer=optimizer_with_scale,
            loss=loss,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            **kwargs,
        )

    def add_featurizers(self, speech_featurizer, text_featurizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def summary(self, line_length=None, **kwargs):
        if self.encoder is not None:
            self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(StreamingTransducer, self).summary(line_length=line_length, **kwargs)

    def train_step(self, batch):
        x, y_true = batch
        with tf.GradientTape() as tape:
            y_pred = self(
                {
                    "input": x["input"],
                    "input_length": x["input_length"],
                    "prediction": x["prediction"],
                    "prediction_length": x["prediction_length"],
                },
                training=True,
            )
            loss = self.loss(y_true, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"train_rnnt_loss": loss}

    def test_step(self, batch):
        x, y_true = batch
        y_pred = self(
            {
                "input": x["input"],
                "input_length": x["input_length"],
                "prediction": x["prediction"],
                "prediction_length": x["prediction_length"],
            },
            training=False,
        )
        loss = self.loss(y_true, y_pred)
        return {"val_rnnt_loss": loss}

    def encoder_inference(self, features: tf.Tensor, states: tf.Tensor):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]
            states (tf.Tensor): previous states of encoders with shape
                [num_rnns, 1 or 2, 1, P]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
            tf.Tensor: states of encoders with shape [num_rnns, 1 or 2, 1, P]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs, new_states = self.encoder.recognize(outputs, states)
            return tf.squeeze(outputs, axis=0), new_states

    def decoder_inference(
        self, encoded: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor
    ):
        """Infer function for decoder

        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence =>
                shape []
            states (nested lists of tf.Tensor): states returned by rnn layers

        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predict_net.recognize(
                predicted, states
            )  # [1, 1, P], states
            ytu = tf.nn.log_softmax(
                self.joint_net([encoded, y], training=False)
            )  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(
        self,
        features: tf.Tensor,
        input_length: tf.Tensor,
        parallel_iterations: int = 10,
        swap_memory: bool = True,
    ):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of padded extracted features

        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded, _ = self.encoder.recognize(features, self.encoder.get_initial_state())
        return self._perform_greedy_batch(
            encoded,
            input_length,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
        )

    def recognize_tflite(self, signal, predicted, encoder_states, prediction_states):
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
            predicted: last predicted character with shape []
            encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
            prediction_states: lastest prediction states with shape
                [num_rnns, 1 or 2, 1, P]

        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype
                tf.int32
            predicted: last predicted character with shape []
            encoder_states: lastest encoder states with shape [num_rnns, 1 or 2, 1, P]
            prediction_states: lastest prediction states with shape
                [num_rnns, 1 or 2, 1, P]
        """
        features = self.speech_featurizer.tf_extract(signal)
        encoded, new_encoder_states = self.encoder_inference(features, encoder_states)
        hypothesis = self._perform_greedy(
            encoded, tf.shape(encoded)[0], predicted, prediction_states
        )
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, new_encoder_states, hypothesis.states

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
                    states=self.predict_net.get_initial_state(),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory,
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition,
                body,
                loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations,
                swap_memory=True,
            )

            decoded = pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
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
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
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

    # -------------------------------- TFLITE -------------------------------------

    def make_tflite_function(self):
        return tf.function(
            self.recognize_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(
                    self.encoder.get_initial_state().get_shape(), dtype=tf.float32
                ),
                tf.TensorSpec(
                    self.predict_net.get_initial_state().get_shape(), dtype=tf.float32
                ),
            ],
        )


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs):
        return merge_two_last_dims(inputs)


class StreamingTransducerBlock(tf.keras.Model):
    def __init__(
        self,
        reduction_factor: int = 0,
        dmodel: int = 640,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(StreamingTransducerBlock, self).__init__(**kwargs)

        if reduction_factor > 0:
            self.reduction = TimeReduction(
                reduction_factor, name=f"{self.name}_reduction"
            )
        else:
            self.reduction = None

        RNN = get_rnn(rnn_type)
        self.rnn = RNN(
            units=rnn_units,
            return_sequences=True,
            name=f"{self.name}_{rnn_type}",
            return_state=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        if layer_norm:
            self.ln = tf.keras.layers.LayerNormalization(name=f"{self.name}_ln")
        else:
            self.ln = None

        self.projection = tf.keras.layers.Dense(
            dmodel,
            name=f"{self.name}_projection",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        if self.reduction is not None:
            outputs = self.reduction(outputs)
        outputs = self.rnn(outputs, training=training)
        outputs = outputs[0]
        if self.ln is not None:
            outputs = self.ln(outputs, training=training)
        outputs = self.projection(outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        outputs = inputs
        if self.reduction is not None:
            outputs = self.reduction(outputs)
        outputs = self.rnn(outputs, training=False, initial_state=states)
        new_states = tf.stack(outputs[1:], axis=0)
        outputs = outputs[0]
        if self.ln is not None:
            outputs = self.ln(outputs, training=False)
        outputs = self.projection(outputs, training=False)
        return outputs, new_states

    def get_config(self):
        conf = {}
        if self.reduction is not None:
            conf.update(self.reduction.get_config())
        conf.update(self.rnn.get_config())
        if self.ln is not None:
            conf.update(self.ln.get_config())
        conf.update(self.projection.get_config())
        return conf


class StreamingTransducerEncoder(tf.keras.Model):
    def __init__(
        self,
        reductions: dict = {0: 3, 1: 2},
        dmodel: int = 640,
        nlayers: int = 8,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super(StreamingTransducerEncoder, self).__init__(**kwargs)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.blocks = [
            StreamingTransducerBlock(
                reduction_factor=reductions.get(
                    i, 0
                ),  # key is index, value is the factor
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}",
            )
            for i in range(nlayers)
        ]

        self.time_reduction_factor = 1
        for i in range(nlayers):
            reduction_factor = reductions.get(i, 0)
            if reduction_factor > 0:
                self.time_reduction_factor *= reduction_factor

    def get_initial_state(self):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, P]
        """
        states = []
        for block in self.blocks:
            states.append(
                tf.stack(
                    block.rnn.get_initial_state(tf.zeros([1, 1, 1], dtype=tf.float32)),
                    axis=0,
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.reshape(inputs)
        for block in self.blocks:
            outputs = block(outputs, training=training, **kwargs)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for encoder network

        Args:
            inputs (tf.Tensor): shape [1, T, F, C]
            states (tf.Tensor): shape [num_lstms, 1 or 2, 1, P]

        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [num_lstms, 1 or 2, 1, P]
        """
        outputs = self.reshape(inputs)
        new_states = []
        for i, block in enumerate(self.blocks):
            outputs, block_states = block.recognize(
                outputs, states=tf.unstack(states[i], axis=0)
            )
            new_states.append(block_states)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.reshape.get_config()
        for block in self.blocks:
            conf.update(block.get_config())
        return conf
