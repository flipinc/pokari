# import collections
import os

import tensorflow as tf
from datasets.dataset import Dataset
from frontends.audio_featurizer import AudioFeaturizer
from frontends.spec_augment import SpectrogramAugmentation
from frontends.text_featurizer import SubwordFeaturizer
from hydra.utils import instantiate
from losses.transducer_loss import TransducerLoss
from metrics.error_rate import ErrorRate
from modules.inference import Inference
from modules.transducer_decoder import TransducerDecoder
from omegaconf import DictConfig, OmegaConf

# Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))


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

        if cfgs.text_feature.subwords and os.path.exists(cfgs.text_feature.subwords):
            print("Loading subwords ...")
            self.text_featurizer = SubwordFeaturizer.load_from_file(
                OmegaConf.to_container(cfgs.text_feature),
                cfgs.text_feature.subwords,
            )
        else:
            print("Generating subwords ...")
            self.text_featurizer = SubwordFeaturizer.build_from_corpus(
                OmegaConf.to_container(cfgs.text_feature),
                cfgs.text_feature.subwords_corpus,
            )
            self.text_featurizer.save_to_file(cfgs.text_feature.subwords)

        self.encoder = instantiate(cfgs.encoder)
        self.predictor = instantiate(
            cfgs.predictor, num_classes=self.text_featurizer.num_classes
        )
        self.joint = instantiate(
            cfgs.joint, num_classes=self.text_featurizer.num_classes
        )

        self.mxp_enabled = cfgs.trainer.mxp
        self.log_interval = cfgs.trainer.log_interval
        self.step_counter = 0

        (
            train_dl,
            train_steps_per_epoch,
            val_dl,
            val_steps_per_epoch,
        ) = self.configure_datasets(
            cfgs.train_ds, cfgs.validation_ds, global_batch_size
        )

        optimizer = self.configure_optimizer(
            optim_cfg=cfgs.optimizer,
            total_steps=train_steps_per_epoch * cfgs.trainer.epochs,
        )

        loss = TransducerLoss(
            blank=self.text_featurizer.blank, global_batch_size=global_batch_size
        )

        self.inference = Inference(
            text_featurizer=self.text_featurizer,
            predictor=self.predictor,
            joint=self.joint,
        )
        self.decoder = TransducerDecoder(labels=[], inference=self.inference)

        self.wer = ErrorRate(kind="wer")
        self.cer = ErrorRate(kind="cer")

        self.compile_args = {
            "optimizer": optimizer,
            "loss": loss,
            "run_eagerly": cfgs.trainer.run_eagerly,
        }

        self.fit_args = {
            "x": train_dl,
            "validation_data": val_dl,
            "steps_per_epoch": train_steps_per_epoch,
            "validation_steps": val_steps_per_epoch,
            "epochs": cfgs.trainer.epochs,
            "workers": cfgs.trainer.workers,
            "max_queue_size": cfgs.trainer.max_queue_size,
            "use_multiprocessing": cfgs.trainer.use_multiprocessing,
            "callbacks": [
                tf.keras.callbacks.ModelCheckpoint(
                    **OmegaConf.to_container(cfgs.trainer.checkpoint)
                ),
                tf.keras.callbacks.experimental.BackupAndRestore(
                    cfgs.trainer.states_dir
                ),
                tf.keras.callbacks.TensorBoard(
                    **OmegaConf.to_container(cfgs.trainer.tensorboard)
                ),
            ],
        }

    def configure_datasets(
        self,
        train_ds_cfg: DictConfig,
        val_ds_cfg: DictConfig,
        global_batch_size: int,
    ):
        train_ds = Dataset(
            audio_featurizer=self.audio_featurizer,
            text_featurizer=self.text_featurizer,
            **OmegaConf.to_container(train_ds_cfg),
        )
        train_dl = train_ds.create(global_batch_size)
        train_steps_per_epoch = train_ds.steps_per_epoch

        val_ds = Dataset(
            audio_featurizer=self.audio_featurizer,
            text_featurizer=self.text_featurizer,
            **OmegaConf.to_container(val_ds_cfg),
        )
        val_dl = val_ds.create(global_batch_size)
        val_steps_per_epoch = val_ds.steps_per_epoch

        return train_dl, train_steps_per_epoch, val_dl, val_steps_per_epoch

    def configure_optimizer(self, optim_cfg: DictConfig, total_steps: int):
        """Configure optimizer

        TODO: support weight decay. the value of weight decay must be decayed too!
            https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW

        """
        optim_cfg = OmegaConf.to_container(optim_cfg)

        lr_scheduler_config = (
            optim_cfg.pop("lr_scheduler") if "lr_scheduler" in optim_cfg else None
        )

        learning_rate = optim_cfg.pop("learning_rate")
        if lr_scheduler_config:
            lr = instantiate(
                lr_scheduler_config,
                total_steps=total_steps,
                learning_rate=learning_rate,
            )
        else:
            lr = learning_rate

        optimizer_type = optim_cfg.pop("name")
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(lr, **optim_cfg)
        else:
            raise NotImplementedError(f"{optimizer} is not yet supported.")

        if self.mxp_enabled:
            tf.print("🐳", tf.__version__)
            # for tensorflow 2.3
            # optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            #     optimizer, "dynamic"
            # )
            # for tensorflow 2.4
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        return optimizer

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

    def _fit(self):
        super(Transducer, self).fit(**self.fit_args)

    def _compile(self):
        super(Transducer, self).compile(**self.compile_args)

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

        encoded_outs, encoded_lens = self.encoder(audio_features, audio_lens)
        decoded_outs = self.predictor(targets, target_lens)

        logits = self.joint([encoded_outs, decoded_outs])

        return {
            "logits": logits,
            "logit_lens": encoded_lens,
            "encoded_outs": encoded_outs,
        }

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
            if self.mxp_enabled:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.mxp_enabled:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        tensorboard_logs = {"train_loss": loss}

        if (self.step_counter + 1) % self.log_interval == 0:
            labels = y_true["labels"]
            encoded_outs = y_pred["encoded_outs"]
            logit_lens = y_pred["logit_lens"]

            results = self.inference._perform_greedy_batch(encoded_outs, logit_lens)

            tf.print("❓ PRED: \n", results[0])
            tf.print(
                "🧩 TRUE: \n",
                tf.strings.unicode_encode(
                    self.text_featurizer.indices2upoints(labels[0]), "UTF-8"
                ),
            )

            self.wer.update_state(results, tf.strings.unicode_encode(labels, "UTF-8"))
            self.cer.update_state(results, tf.strings.unicode_encode(labels, "UTF-8"))

            tensorboard_logs.update({"train_wer": self.wer.result()})
            tensorboard_logs.update({"train_cer": self.cer.result()})

            # Average over each interval
            self.wer.reset_states()
            self.cer.reset_states()

        return tensorboard_logs

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

        tensorboard_logs = {"val_loss": loss}

        return tensorboard_logs

    def encoder_inference(self, audio_features: tf.Tensor, states: tf.Tensor):
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
            outputs, states = self.encoder.recognize(audio_features, states)
            return tf.squeeze(outputs, axis=0), states

    # def decoder_inference(
    #     self, encoded_outs: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor
    # ):
    #     """Infer function for decoder

    #     Args:
    #         encoded_outs (tf.Tensor): output of encoder at each time step => shape [E]
    #         predicted (tf.Tensor): last character index of predicted sequence =>
    #             shape []
    #         states (nested lists of tf.Tensor): states returned by rnn layers

    #     Returns:
    #         (ytu, new_states)
    #     """
    #     with tf.name_scope(f"{self.name}_decoder"):
    #         encoded_outs = tf.reshape(encoded_outs, [1, 1, -1])  # [E] => [1, 1, E]
    #         predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
    #         y, new_states = self.predictor.recognize(
    #             predicted, states
    #         )  # [1, 1, P], states
    #         ytu = tf.nn.log_softmax(
    #             self.joint([encoded_outs, y], training=False)
    #         )  # [1, 1, V]
    #         ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
    #         return ytu, new_states

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

    def recognize_tflite(
        self, audio_signals, predicted, cache_encoder_states, cache_predictor_states
    ):
        audio_signals = tf.expand_dims(audio_signals, axis=0)  # add batch dim
        audio_lens = tf.expand_dims(tf.shape(audio_signals)[1], axis=0)
        audio_features, _ = self.audio_featurizer.tf_extract(audio_signals, audio_lens)
        encoded_outs, cache_encoder_states = self.encoder_inference(
            audio_features, cache_encoder_states
        )
        hypothesis = self._perform_greedy(
            encoded_outs, tf.shape(encoded_outs)[0], predicted, cache_predictor_states
        )
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, cache_encoder_states, hypothesis.states

    # def _perform_greedy_batch(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = False,
    #     version: str = "v1",
    # ):
    #     with tf.name_scope(f"{self.name}_perform_greedy_batch"):
    #         total_batch = tf.shape(encoded)[0]
    #         batch = tf.constant(0, dtype=tf.int32)

    #         t_max = tf.math.reduce_max(encoded_length)

    #         greedy_fn = (
    #             self._perform_greedy if version == "v1" else self._perform_greedy_v2
    #         )

    #         decoded = tf.TensorArray(
    #             dtype=tf.int32,
    #             size=total_batch,
    #             dynamic_size=False,
    #             clear_after_read=False,
    #             element_shape=tf.TensorShape([None]),
    #         )

    #         def condition(batch, _):
    #             return tf.less(batch, total_batch)

    #         def body(batch, decoded):
    #             hypothesis = greedy_fn(
    #                 encoded=encoded[batch],
    #                 encoded_length=encoded_length[batch],
    #                 predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
    #                 states=self.predictor.get_initial_state(batch_size=1),
    #                 parallel_iterations=parallel_iterations,
    #                 swap_memory=swap_memory,
    #             )
    #             prediction = tf.pad(
    #                 hypothesis.prediction,
    #                 paddings=[[0, t_max - tf.shape(hypothesis.prediction)[0]]],
    #                 mode="CONSTANT",
    #                 constant_values=self.text_featurizer.blank,
    #             )
    #             decoded = decoded.write(batch, prediction)
    #             return batch + 1, decoded

    #         batch, decoded = tf.while_loop(
    #             condition,
    #             body,
    #             loop_vars=[batch, decoded],
    #             parallel_iterations=parallel_iterations,
    #             swap_memory=True,
    #         )

    #         return self.text_featurizer.iextract(decoded.stack())

    # def _perform_greedy(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     predicted: tf.Tensor,
    #     states: tf.Tensor,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = False,
    # ):
    #     with tf.name_scope(f"{self.name}_greedy"):
    #         time = tf.constant(0, dtype=tf.int32)
    #         total = encoded_length

    #         hypothesis = Hypothesis(
    #             index=predicted,
    #             prediction=tf.TensorArray(
    #                 dtype=tf.int32,
    #                 size=total,
    #                 dynamic_size=False,
    #                 clear_after_read=False,
    #                 element_shape=tf.TensorShape([]),
    #             ),
    #             states=states,
    #         )

    #         def condition(_time, _hypothesis):
    #             return tf.less(_time, total)

    #         def body(_time, _hypothesis):
    #             ytu, _states = self.decoder_inference(
    #                 # avoid using [index] in tflite
    #                 encoded_outs=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
    #                 predicted=_hypothesis.index,
    #                 states=_hypothesis.states,
    #             )
    #             _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

    #             # something is wrong with tflite that drop support for tf.cond
    #             # def equal_blank_fn(): return _hypothesis.index, _hypothesis.states
    #             # def non_equal_blank_fn(): return _predict, _states  # update if the
    #             # new prediction is a non-blank
    #             # _index, _states = tf.cond(tf.equal(_predict, blank), equal_blank_fn,
    #             # non_equal_blank_fn)

    #             _equal = tf.equal(_predict, self.text_featurizer.blank)
    #             _index = tf.where(_equal, _hypothesis.index, _predict)
    #             _states = tf.where(_equal, _hypothesis.states, _states)

    #             _prediction = _hypothesis.prediction.write(_time, _predict)
    #             _hypothesis = Hypothesis(
    #                 index=_index, prediction=_prediction, states=_states
    #             )

    #             return _time + 1, _hypothesis

    #         time, hypothesis = tf.while_loop(
    #             condition,
    #             body,
    #             loop_vars=[time, hypothesis],
    #             parallel_iterations=parallel_iterations,
    #             swap_memory=swap_memory,
    #         )

    #         return Hypothesis(
    #             index=hypothesis.index,
    #             prediction=hypothesis.prediction.stack(),
    #             states=hypothesis.states,
    #         )

    # def _perform_greedy_v2(
    #     self,
    #     encoded: tf.Tensor,
    #     encoded_length: tf.Tensor,
    #     predicted: tf.Tensor,
    #     states: tf.Tensor,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = False,
    # ):
    #     """ Ref: https://arxiv.org/pdf/1801.00841.pdf """
    #     with tf.name_scope(f"{self.name}_greedy_v2"):
    #         time = tf.constant(0, dtype=tf.int32)
    #         total = encoded_length

    #         hypothesis = Hypothesis(
    #             index=predicted,
    #             prediction=tf.TensorArray(
    #                 dtype=tf.int32,
    #                 size=0,
    #                 dynamic_size=True,
    #                 clear_after_read=False,
    #                 element_shape=tf.TensorShape([]),
    #             ),
    #             states=states,
    #         )

    #         def condition(_time, _hypothesis):
    #             return tf.less(_time, total)

    #         def body(_time, _hypothesis):
    #             ytu, _states = self.decoder_inference(
    #                 # avoid using [index] in tflite
    #                 encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
    #                 predicted=_hypothesis.index,
    #                 states=_hypothesis.states,
    #             )
    #             _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

    #             _equal = tf.equal(_predict, self.text_featurizer.blank)
    #             _index = tf.where(_equal, _hypothesis.index, _predict)
    #             _states = tf.where(_equal, _hypothesis.states, _states)
    #             _time = tf.where(_equal, _time + 1, _time)

    #             _prediction = _hypothesis.prediction.write(_time, _predict)
    #             _hypothesis = Hypothesis(
    #                 index=_index, prediction=_prediction, states=_states
    #             )

    #             return _time, _hypothesis

    #         time, hypothesis = tf.while_loop(
    #             condition,
    #             body,
    #             loop_vars=[time, hypothesis],
    #             parallel_iterations=parallel_iterations,
    #             swap_memory=swap_memory,
    #         )

    #         return Hypothesis(
    #             index=hypothesis.index,
    #             prediction=hypothesis.prediction.stack(),
    #             states=hypothesis.states,
    #         )

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
                    self.predictor.get_initial_state(batch_size=1).get_shape(),
                    dtype=tf.float32,
                ),
            ],
        )
