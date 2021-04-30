from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from datasets.dataset import Dataset
from frontends.audio_featurizer import AudioFeaturizer
from frontends.spec_augment import SpectrogramAugmentation
from hydra.utils import instantiate
from metrics.error_rate import ErrorRate
from omegaconf import DictConfig, OmegaConf


class BaseModel(tf.keras.Model):
    def __init__(
        self,
        loss_module,
        cfgs: DictConfig,
        global_batch_size: int,
        setup_training: bool = True,
    ):
        super().__init__()

        self.setup_training = setup_training

        self.audio_featurizer = AudioFeaturizer(
            **OmegaConf.to_container(cfgs.audio_feature)
        )
        self.text_featurizer = instantiate(cfgs.text_feature)

        self.spec_augment = SpectrogramAugmentation(
            **OmegaConf.to_container(cfgs.spec_augment)
        )

        # since in tensorflow 2.3 keras.metrics are registered as layers, models that
        # were trained in tensorflow 2.4 cannot be loaded. To avoid issues like this,
        # it is better to just disable all training logics when they are unused.
        # And also this is useful to remove unnecessary things from the production model
        if self.setup_training:
            self.step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

            self.wer = ErrorRate(kind="wer")
            self.cer = ErrorRate(kind="cer")

            self.wer_value = tf.Variable(100.0, dtype=tf.float32, trainable=False)
            self.cer_value = tf.Variable(100.0, dtype=tf.float32, trainable=False)

            self.debugging = (
                cfgs.trainer.debugging if "debugging" in cfgs.trainer else None
            )
            self.mxp_enabled = cfgs.trainer.mxp if "mxp" in cfgs.trainer else None
            self.log_interval = (
                cfgs.trainer.log_interval if "log_interval" in cfgs.trainer else None
            )

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

            loss = loss_module(
                blank=self.text_featurizer.blank, global_batch_size=global_batch_size
            )

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
            # TODO: not every learning rate scheduler supports warmup steps
            self.warmup_steps = lr.warmup_steps
        else:
            lr = learning_rate

        optimizer_type = optim_cfg.pop("name")
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(lr, **optim_cfg)
        elif optimizer_type == "adamw":
            # TODO: the value of weight decay must be decayed too!
            # https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW
            optimizer = tfa.optimizers.AdamW(learning_rate=lr, **optim_cfg)
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
                NotImplementedError("😊 Hello There!!")

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

        self._summary(line_length=150)

    def _fit(self):
        super(BaseModel, self).fit(**self.fit_args)

    def _compile(self):
        super(BaseModel, self).compile(**self.compile_args)

    def _summary(self, **kwargs):
        """
        Summarization can be only done at a tf.keras.Model level because
        tf.keras.layers.Layer do not support .summary(). You could use
        tf.keras.Model for the underlying layers, but it will not be serializable.
        """
        super(BaseModel, self).summary(**kwargs)

    def _save(self, **kwargs):
        """

        TODO: Right now, only h5 models are saved periodically. In order to save a
        model in SavedModel format, the saved model needs to be loaded once

        """
        super(BaseModel, self).save(**kwargs)

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

        gradients = self.on_gradient_computed(gradients)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        learning_rate = (
            self.optimizer.lr.value
            if hasattr(
                self, "warmup_steps"
            )  # warm step is set if lr_scheduler is enabled
            else self.optimizer.lr
        )

        tensorboard_logs = {
            "loss": loss,
            "lr": learning_rate,
            # this is useful for replaying training from mid-point using a saved model
            # and lr_scheduler
            "step": self.step_counter,
        }

        logs = self.on_step_end(
            labels=y_true["labels"],
            encoded_outs=y_pred["encoded_outs"],
            logits=y_pred["logits"],
            logit_lens=y_pred["logit_lens"],
        )
        if logs is not None:
            tensorboard_logs.update(logs)

        self.step_counter.assign_add(1)

        return tensorboard_logs

    # keras `train_step` is run in graph mode, however, autograph feature is turned off.
    # in order to use autograph feature, explicitly calling tf.function is required.
    # this is why this function is separated from `train_step`
    # ref: https://github.com/tensorflow/tensorflow/issues/42119#issuecomment-747098474
    @tf.function
    def on_gradient_computed(self, gradients):
        if self.gradient_clip_val is not None:
            high = self.gradient_clip_val
            low = self.gradient_clip_val * -1
            gradients = [(tf.clip_by_value(grad, low, high)) for grad in gradients]

        # add variational noise for better generalization to out-of-domain data
        # ref: https://arxiv.org/pdf/2005.03271v1.pdf
        if self.variational_noise_cfg is not None:
            start_step = self.variational_noise_cfg["start_step"]

            if tf.less_equal(start_step, self.step_counter):
                new_gradients = []

                for grad in gradients:
                    values = tf.convert_to_tensor(
                        grad.values if isinstance(grad, tf.IndexedSlices) else grad
                    )

                    noise = tf.random.normal(
                        tf.shape(values),
                        mean=self.variational_noise_cfg.get("mean", 0),
                        stddev=self.variational_noise_cfg.get("stddev", 0.01),
                        dtype=values.dtype,
                    )
                    values = tf.add(grad, noise)

                    if isinstance(grad, tf.IndexedSlices):
                        values = tf.IndexedSlices(
                            values, grad.indices, grad.dense_shape
                        )

                    new_gradients.append(values)

                gradients = new_gradients

        if self.debugging:
            tf.print("Checking gradients value...")
            for grad in gradients:
                tf.debugging.check_numerics(grad, "Gradients have invalid value!!")

        return gradients

    def on_step_end(self, labels, logits, logit_lens, encoded_outs):
        raise NotImplementedError()

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
