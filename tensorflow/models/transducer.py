import os
from datetime import datetime

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from datasets.dataset import Dataset
from frontends.audio_featurizer import AudioFeaturizer
from frontends.spec_augment import SpectrogramAugmentation
from hydra.utils import instantiate
from losses.transducer_loss import TransducerLoss
from metrics.error_rate import ErrorRate
from modules.inference import Inference
from modules.mock_stream import MockStream
from modules.transducer_decoder import TransducerDecoder
from omegaconf import DictConfig, OmegaConf


class Transducer(tf.keras.Model):
    def __init__(
        self,
        cfgs: DictConfig,
        global_batch_size: int,
        setup_training: bool = True,
    ):
        super().__init__()

        self.audio_featurizer = AudioFeaturizer(
            **OmegaConf.to_container(cfgs.audio_feature)
        )
        self.spec_augment = SpectrogramAugmentation(
            **OmegaConf.to_container(cfgs.spec_augment)
        )

        self.text_featurizer = instantiate(cfgs.text_feature)

        self.encoder = instantiate(cfgs.encoder)
        self.predictor = instantiate(
            cfgs.predictor, num_classes=self.text_featurizer.num_classes
        )
        self.joint = instantiate(
            cfgs.joint, num_classes=self.text_featurizer.num_classes
        )

        self.inference = Inference(
            text_featurizer=self.text_featurizer,
            predictor=self.predictor,
            joint=self.joint,
        )
        self.decoder = TransducerDecoder(labels=[], inference=self.inference)

        # since in tensorflow 2.3 keras.metrics are registered as layers, models that
        # were trained in tensorflow 2.4 cannot be loaded. To avoid issues like this,
        # it is better to just disable all training logics when they are unused
        if setup_training:
            self.debugging = (
                cfgs.trainer.debugging if "debugging" in cfgs.trainer else None
            )
            self.mxp_enabled = cfgs.trainer.mxp if "mxp" in cfgs.trainer else None
            self.log_interval = (
                cfgs.trainer.log_interval if "log_interval" in cfgs.trainer else None
            )
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

            self.mock_stream = MockStream(
                audio_featurizer=self.audio_featurizer,
                text_featurizer=self.text_featurizer,
                encoder=self.encoder,
                predictor=self.predictor,
                inference=self.inference,
            )

            self.wer = ErrorRate(kind="wer")
            self.cer = ErrorRate(kind="cer")

            self.compile_args = {
                "optimizer": optimizer,
                "loss": loss,
                "run_eagerly": cfgs.trainer.run_eagerly,
            }

            now = datetime.now().strftime("%Y%m%d-%H%M%S")

            tensorboard_cfg = OmegaConf.to_container(cfgs.trainer.tensorboard)
            tensorboard_cfg.update({"log_dir": tensorboard_cfg["log_dir"] + now})

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
                    tf.keras.callbacks.TensorBoard(**tensorboard_cfg),
                ],
            }

    def configure_datasets(
        self,
        train_ds_cfg: DictConfig,
        val_ds_cfg: DictConfig,
        global_batch_size: int,
    ):
        self.train_ds = Dataset(
            audio_featurizer=self.audio_featurizer,
            text_featurizer=self.text_featurizer,
            **OmegaConf.to_container(train_ds_cfg),
        )
        train_dl = self.train_ds.create(global_batch_size)
        train_steps_per_epoch = self.train_ds.steps_per_epoch

        self.val_ds = Dataset(
            audio_featurizer=self.audio_featurizer,
            text_featurizer=self.text_featurizer,
            **OmegaConf.to_container(val_ds_cfg),
        )
        val_dl = self.val_ds.create(global_batch_size)
        val_steps_per_epoch = self.val_ds.steps_per_epoch

        return train_dl, train_steps_per_epoch, val_dl, val_steps_per_epoch

    def configure_optimizer(self, optim_cfg: DictConfig, total_steps: int):
        """Configure optimizer"""
        optim_cfg = OmegaConf.to_container(optim_cfg)

        self.gradient_clip_val = (
            optim_cfg.pop("gradient_clip_val")
            if "gradient_clip_val" in optim_cfg
            else None
        )

        self.variational_noise_cfg = (
            optim_cfg.pop("variational_noise")
            if "variational_noise" in optim_cfg
            else None
        )

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
            self.warmup_steps = lr.warmup_steps
        else:
            lr = learning_rate

        optimizer_type = optim_cfg.pop("name")
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(lr, **optim_cfg)
        elif optimizer_type == "adamw":
            # TODO: For weight decay, the value of weight decay must be decayed too!
            # https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW
            optimizer = tfa.optimizers.AdamW(learning_rate=lr, **optim_cfg)
        else:
            raise NotImplementedError(f"{optimizer} is not yet supported.")

        if self.mxp_enabled:
            # TFLite conversion does not work for tf version 2.4 and RTX3090 training
            # does not work for tf version 2.3
            if "2.3" in tf.__version__:
                optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, "dynamic"
                )
            elif "2.4" in tf.__version__:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            else:
                NotImplementedError(
                    "Please check if this version is runnable and tflite convertible."
                )

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
        if self.log_interval == 1 and self.compile_args["run_eagerly"] is False:
            raise ValueError("Logging with Graph mode is not supported")

        super(Transducer, self).compile(**self.compile_args)

    def call(self, inputs, training=False):
        audio_signals = inputs["audio_signals"]
        audio_lens = inputs["audio_lens"]
        targets = inputs["targets"]
        target_lens = inputs["target_lens"]

        # [B, T, n_mels]
        audio_features, audio_lens = self.audio_featurizer(audio_signals, audio_lens)

        # [B, T, n_mels]
        if training:
            audio_features = self.spec_augment(audio_features)

        # [B, T, D_e]
        encoded_outs, encoded_lens = self.encoder(audio_features, audio_lens)

        # [B, U, D_p]
        decoded_outs = self.predictor(targets, target_lens)

        # [B, T, U, D_j]
        logits = self.joint([encoded_outs, decoded_outs])

        return {
            "logits": logits,
            "logit_lens": encoded_lens,
            "encoded_outs": encoded_outs,
        }

    # in order to use AutoGraph feature (control flow conversions and so on),
    # explicitly calling tf.function is required. This will be useful when implementing
    # variational noise in the future
    # ref: https://github.com/tensorflow/tensorflow/issues/42119#issuecomment-747098474
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

        # TODO: variational noise (gaussian noise) is does not work. Copy simulate()
        # in pytorch implementation

        # if self.variational_noise_cfg is not None:
        #     start_step = self.variational_noise_cfg.pop("start_step")

        #     if tf.equal(start_step, -1):
        #         start_step = self.warmup_steps

        #     if tf.less_equal(start_step, self.step_counter):
        #         gradients = [
        #             tf.add(
        #                 grad,
        #                 tf.random.normal(
        #                     tf.shape(grad),
        #                     **self.variational_noise_cfg,
        #                     dtype=grad.dtype,
        #                 ),
        #             )
        #             for grad in gradients
        #         ]

        if self.gradient_clip_val is not None:
            high = self.gradient_clip_val
            low = self.gradient_clip_val * -1
            gradients = [(tf.clip_by_value(grad, low, high)) for grad in gradients]

        if self.debugging:
            tf.print("Checking gradients value...")
            for grad in gradients:
                tf.debugging.check_numerics(grad, "Gradients have invalid value!!")

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        tensorboard_logs = {"loss": loss}

        # TODO: This code can be executed only in eager mode. so for graph mode,
        # log_interval must be > 1 for tracing to be disabled
        if (
            self.log_interval is not None
            and (self.step_counter + 1) % self.log_interval == 0
        ):
            labels = y_true["labels"]
            encoded_outs = y_pred["encoded_outs"]
            logit_lens = y_pred["logit_lens"]

            results, _, _ = self.inference.greedy_batch_decode(encoded_outs, logit_lens)

            tf.print("❓ PRED: \n", results[0])
            tf.print(
                "🧩 TRUE: \n",
                tf.strings.unicode_encode(
                    self.text_featurizer.indices2upoints(labels[0]), "UTF-8"
                ),
            )

            self.wer.update_state(results, tf.strings.unicode_encode(labels, "UTF-8"))
            self.cer.update_state(results, tf.strings.unicode_encode(labels, "UTF-8"))

            tensorboard_logs.update({"wer": self.wer.result()})
            tensorboard_logs.update({"cer": self.cer.result()})

            # Average over each step
            self.wer.reset_states()
            self.cer.reset_states()

        self.step_counter += 1

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

        tensorboard_logs = {"loss": loss}

        return tensorboard_logs

    def stream(
        self,
        manifest_idx: int = 0,
        enable_graph: bool = False,
    ):
        """Mock streaming on CPU by segmenting an audio into chunks

        Args:
            manifest_idx: Audio path and transcript pair to use for mock streaming.
            enable_graph: Enable graph mode and hide intermediate print outputs

        """
        path, transcript = self.train_ds.entries[manifest_idx]
        tf.print(f"🎙 Using {path}...")

        audio_signal, native_rate = librosa.load(
            os.path.expanduser(path),
            sr=self.audio_featurizer.sample_rate,
            mono=True,
            dtype=np.float32,
        )
        audio_signal = tf.convert_to_tensor(audio_signal)
        transcript = tf.strings.unicode_encode(
            self.text_featurizer.indices2upoints(
                tf.strings.to_number(tf.strings.split(transcript), out_type=tf.int32)
            ),
            output_encoding="UTF-8",
        )

        if enable_graph:
            self.fn = tf.function(self.mock_stream)
        else:
            self.fn = self.mock_stream

        self.fn(audio_signal)

        tf.print("💎: ", transcript)

    def stream_batch_tflite(
        self, audio_signals, prev_tokens, cache_encoder_states, cache_predictor_states
    ):
        audio_lens = tf.expand_dims(tf.shape(audio_signals)[1], axis=0)
        audio_features, _ = self.audio_featurizer(audio_signals, audio_lens)

        encoded_outs, cache_encoder_states = self.encoder.stream(
            audio_features, cache_encoder_states
        )

        (
            predictions,
            prev_tokens,
            cache_predictor_states,
        ) = self.inference.greedy_batch_decode(
            encoded_outs=encoded_outs,
            encoded_lens=tf.shape(encoded_outs)[1],
            prev_tokens=prev_tokens,
            cache_states=cache_predictor_states,
        )

        transcripts = self.text_featurizer.indices2upoints(predictions)

        return transcripts, prev_tokens, cache_encoder_states, cache_predictor_states

    def stream_one_tflite(
        self, audio_signal, prev_token, cache_encoder_states, cache_predictor_states
    ):
        """Streaming tflite model for batch size = 1

        Args:
            audio_signal: [T]
            prev_token: [1]
            cache_encoder_states: size depends on encoder type
            cache_predictor_states: [N, 2, B, D_p]

        """
        audio_signal = tf.expand_dims(audio_signal, axis=0)  # add batch dim
        audio_len = tf.expand_dims(tf.shape(audio_signal)[1], axis=0)
        audio_feature, _ = self.audio_featurizer(
            audio_signal, audio_len, training=False, inference=True
        )

        encoded_out, cache_encoder_states = self.encoder.stream(
            audio_feature, cache_encoder_states
        )
        encoded_out = tf.squeeze(encoded_out, axis=0)  # remove batch dim

        hypothesis = self.inference.greedy_decode(
            encoded_out=encoded_out,
            encoded_len=tf.shape(encoded_out)[0],
            prev_token=prev_token,
            cache_states=cache_predictor_states,
        )

        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)

        return transcript, hypothesis.index, cache_encoder_states, hypothesis.states

    def make_one_tflite_function(self):
        return tf.function(
            self.stream_one_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(
                    self.encoder.get_initial_state(batch_size=1).get_shape(),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    self.predictor.get_initial_state(batch_size=1).get_shape(),
                    dtype=tf.float32,
                ),
            ],
        )
