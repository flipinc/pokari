import collections
import os

import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras import mixed_precision as mxp

from audio_featurizer import AudioFeaturizer
from dataset import Dataset
from spec_augment import SpectrogramAugmentation
from text_featurizer import SubwordFeaturizer
from transducer_loss import TransducerLoss
from utils import pad_prediction_tfarray

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


class Transducer(tf.keras.Model):
    def __init__(
        self,
        cfgs: DictConfig,
        global_batch_size: int,
    ):
        super().__init__()

        self.audio_featurizer = AudioFeaturizer(
            **OmegaConf.to_container(cfgs.audio_feature)
        )
        self.spec_augment = SpectrogramAugmentation(
            **OmegaConf.to_container(cfgs.spec_augment)
        )

        if cfgs.subwords and os.path.exists(cfgs.subwords):
            print("Loading subwords ...")
            text_featurizer = SubwordFeaturizer.load_from_file(
                OmegaConf.to_container(cfgs.decoder_config),
                cfgs.subwords,
            )
        else:
            print("Generating subwords ...")
            text_featurizer = SubwordFeaturizer.build_from_corpus(
                OmegaConf.to_container(cfgs.decoder_config),
                cfgs.subwords_corpus,
            )
            text_featurizer.save_to_file(cfgs.subwords)

        vocab_size = text_featurizer.num_classes

        train_dataset = Dataset(
            speech_featurizer=self.audio_featurizer,
            text_featurizer=text_featurizer,
            **OmegaConf.to_container(cfgs.learning_config.train_dataset_config),
        )
        self.train_steps = train_dataset.total_steps
        self.train_dl = train_dataset.create(global_batch_size)

        eval_dataset = Dataset(
            speech_featurizer=self.audio_featurizer,
            text_featurizer=text_featurizer,
            **OmegaConf.to_container(cfgs.learning_config.eval_dataset_config),
        )
        self.val_dl = eval_dataset.create(global_batch_size)

        self.encoder = instantiate(cfgs.encoder)
        self.predictor = instantiate(cfgs.predictor, vocab_size=vocab_size)
        self.joint = instantiate(cfgs.joint, vocab_size=vocab_size)

        optimizer = tf.keras.optimizers.get(
            OmegaConf.to_container(cfgs.learning_config.optimizer_config)
        )
        self.compile(
            optimizer=optimizer,
            global_batch_size=global_batch_size,
            blank=text_featurizer.blank,
        )

        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                **OmegaConf.to_container(cfgs.learning_config.running_config.checkpoint)
            ),
            tf.keras.callbacks.experimental.BackupAndRestore(
                cfgs.learning_config.running_config.states_dir
            ),
            tf.keras.callbacks.TensorBoard(
                **OmegaConf.to_container(
                    cfgs.learning_config.running_config.tensorboard
                )
            ),
        ]

        self.num_epochs = cfgs.learning_config.running_config.num_epochs

    def _build(self):
        self(
            {
                "audio_signals": tf.keras.Input(shape=[None], dtype=tf.float32),
                "audio_lens": tf.keras.Input(shape=[], dtype=tf.int32),
                "targets": tf.keras.Input(shape=[None], dtype=tf.int32),
                "target_lens": tf.keras.Input(shape=[], dtype=tf.int32),
            },
            training=True,
        )

        self.summary(line_length=150)

    def call(self, inputs, training=False):
        audio_signals = inputs["audio_signals"]
        audio_lens = inputs["audio_lens"]
        targets = inputs["targets"]
        target_lens = inputs["target_lens"]

        # [B, T, n_mels]
        audio_features, audio_lens = self.audio_featurizer.tf_extract(
            audio_signals, audio_lens
        )

        # [B, T, n_mels]
        if training:
            audio_features = self.spec_augment(audio_features)

        encoded_outs, audio_lens = self.encoder(audio_features, audio_lens)
        decoded_outs = self.predictor(targets, target_lens)

        logits = self.joint([encoded_outs, decoded_outs])

        return {"logits": logits, "logit_lens": audio_lens}

    def train(self):
        super(Transducer, self).fit(
            self.train_dl,
            epochs=self.num_epochs,
            validation_data=self.val_dl,
            callbacks=self.callbacks,
            steps_per_epoch=self.train_steps,
        )

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
        loss = TransducerLoss(blank=blank, global_batch_size=global_batch_size)
        optimizer_with_scale = mxp.experimental.LossScaleOptimizer(
            tf.keras.optimizers.get(optimizer), "dynamic"
        )
        super(Transducer, self).compile(
            optimizer=optimizer_with_scale,
            loss=loss,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=True,
            **kwargs,
        )

    def train_step(self, batch):
        x, y_true = batch
        with tf.GradientTape() as tape:
            y_pred = self(
                {
                    "audio_signals": x["audio_signals"],
                    "audio_lens": x["audio_lens"],
                    "targets": x["targets"],
                    "target_lens": x["target_lens"],
                },
                training=True,
            )
            loss = self.loss(y_true, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"train_loss": loss}

    def test_step(self, batch):
        x, y_true = batch
        y_pred = self(
            {
                "audio_signals": x["audio_signals"],
                "audio_lens": x["audio_lens"],
                "targets": x["targets"],
                "target_lens": x["target_lens"],
            },
            training=False,
        )
        loss = self.loss(y_true, y_pred)
        return {"val_loss": loss}

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
            y, new_states = self.predictor.recognize(
                predicted, states
            )  # [1, 1, P], states
            ytu = tf.nn.log_softmax(
                self.joint([encoded, y], training=False)
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
                    states=self.predictor.get_initial_state(),
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
                    self.predictor.get_initial_state().get_shape(), dtype=tf.float32
                ),
            ],
        )
