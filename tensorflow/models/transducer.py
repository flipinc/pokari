from datetime import datetime

import tensorflow as tf
from datasets.dataset import Dataset
from frontends.audio_featurizer import AudioFeaturizer
from frontends.spec_augment import SpectrogramAugmentation
from hydra.utils import instantiate
from losses.transducer_loss import TransducerLoss
from metrics.error_rate import ErrorRate
from modules.inference import Inference
from modules.transducer_decoder import TransducerDecoder
from omegaconf import DictConfig, OmegaConf


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

        self.text_featurizer = instantiate(cfgs.text_feature)

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
        else:
            raise NotImplementedError(f"{optimizer} is not yet supported.")

        if self.mxp_enabled:
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

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        tensorboard_logs = {"loss": loss}

        # TODO: This code can be executed only in eager mode. so for graph mode,
        # log_interval must be > 1 for tracing to be disabled
        if (self.step_counter + 1) % self.log_interval == 0:
            labels = y_true["labels"]
            encoded_outs = y_pred["encoded_outs"]
            logit_lens = y_pred["logit_lens"]

            results, _ = self.inference.greedy_batch_decode(encoded_outs, logit_lens)

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

            # Average over each interval
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

    # TODO: this function is not implemented for new APIs yet
    # @tf.function
    # def stream(
    #     self,
    #     features: tf.Tensor,
    #     input_length: tf.Tensor,
    #     parallel_iterations: int = 10,
    #     swap_memory: bool = True,
    # ):
    #     encoded, _ = self.encoder.stream(features, self.encoder.get_initial_state())
    #     return self.inference.greedy_naive_batch_decode(
    #         encoded,
    #         input_length,
    #         parallel_iterations=parallel_iterations,
    #         swap_memory=swap_memory,
    #     )

    def stream_one_tflite(
        self, audio_signals, predicted, cache_encoder_states, cache_predictor_states
    ):
        """Streaming tflite model for batch size = 1

        TODO: Add batch_size as an argument and if its one, implement the following.

        """
        audio_signals = tf.expand_dims(audio_signals, axis=0)  # add batch dim
        audio_lens = tf.expand_dims(tf.shape(audio_signals)[1], axis=0)
        audio_features, _ = self.audio_featurizer.tf_extract(audio_signals, audio_lens)

        with tf.name_scope(f"{self.name}_encoder"):
            encoded_outs, cache_encoder_states = self.encoder.stream(
                audio_features, cache_encoder_states
            )
            encoded_outs = tf.squeeze(encoded_outs, axis=0)  # remove batch dim

        hypothesis = self.inference.greedy_decode(
            encoded_outs, tf.shape(encoded_outs)[0], predicted, cache_predictor_states
        )
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return transcript, hypothesis.index, cache_encoder_states, hypothesis.states

    def make_tflite_function(self):
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
