import tensorflow as tf
from datasets.audio_augment import get_augmentations
from datasets.audio_to_text import DatasetCreator
from hydra.utils import instantiate
from losses.transducer import TransducerLoss
from omegaconf import DictConfig, OmegaConf


class Transducer(tf.keras.Model):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.labels = cfg.labels

        self.train_dl, self.num_train_samples = self.setup_dataloader_from_config(
            cfg.train_ds
        )
        self.validation_dl, self.num_val_samples = self.setup_dataloader_from_config(
            cfg.validation_ds
        )

        if self.num_train_samples is None:
            if "num_samples" in cfg.train_ds:
                self.num_train_samples = cfg.train_ds.num_samples
            else:
                raise ValueError(
                    "num_samples cannot be computed when using custom tfrecord"
                )

        if self.num_val_samples is None:
            if "num_samples" in cfg.validation_ds:
                self.num_val_samples = cfg.validation_ds.num_samples
            else:
                raise ValueError(
                    "num_samples cannot be computed when using custom tfrecord"
                )

        print(
            f"Using {self.num_train_samples} training samples, "
            f"and {self.num_val_samples} validation samples"
        )

        self.preprocessor = instantiate(cfg.preprocessor)
        self.spec_augment = instantiate(cfg.spec_augment)

        self.encoder = instantiate(cfg.encoder)
        self.predictor = instantiate(cfg.predictor, vocab_size=len(self.labels))

        self.joint = instantiate(
            cfg.joint,
            encoder_hidden=cfg.encoder.dim_model,
            predictor_hidden=cfg.predictor.dim_model,
            vocab_size=len(self.labels),
        )

        # self.inference = instantiate(
        #     cfg.inference,
        #     predictor=self.predictor,
        #     joint=self.joint,
        #     blank_index=len(self.labels),
        # )

        # self.decoder = TransducerDecoder(labels=self.labels, inference=self.inference)

        # self.stream = instantiate(
        #     cfg.stream,
        #     preprocessor=self.preprocessor,
        #     encoder=self.encoder,
        #     decoder=self.decoder,
        # )

        # self.wer = TransducerWER(
        #     decoder=self.decoder,
        #     batch_dim_index=0,
        #     use_cer=False,
        #     log_prediction=True,
        #     dist_sync_on_step=True,
        # )

        loss = TransducerLoss(
            batch_size=cfg.train_ds.batch_size, vocab_size=len(self.labels)
        )

        optim_cfg = OmegaConf.to_container(cfg.optimizer)

        self.train_steps = self.num_train_samples // cfg.train_ds.batch_size
        self.validation_steps = self.num_val_samples // cfg.validation_ds.batch_size
        total_steps = self.train_steps * cfg.trainer.epochs

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

        # TODO: support weight decay. the value of weight decay must be decayed too!
        # ref: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW

        optimizer_type = optim_cfg.pop("name")
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(lr, **optim_cfg)
        else:
            raise NotImplementedError(f"{optimizer} is not yet supported.")

        self.mixed_precision_enabled = (
            "precision" in cfg and cfg["precision"] is not None
        )

        if self.mixed_precision_enabled:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        super(Transducer, self).compile(
            optimizer=optimizer, loss=loss, run_eagerly=cfg.trainer.run_eagerly
        )

        self.trainer_cfg = OmegaConf.to_container(cfg.trainer)
        self.postprocessor_cfg = OmegaConf.to_container(cfg.postprocessor)

    def setup_dataloader_from_config(self, cfg: DictConfig):
        augmentor = get_augmentations(cfg["augmentor"]) if "augmentor" in cfg else None
        dataset_creator = DatasetCreator(
            labels=self.labels,
            stage=cfg["stage"],
            batch_size=cfg["batch_size"],
            tfrecords_dir=cfg["tfrecords_dir"],
            tfrecords_shards=cfg["tfrecords_shards"],
            cache=cfg["cache"],
            shuffle=cfg["shuffle"],
            drop_remainder=cfg.get("drop_remainder", False),
            sample_rate=cfg["sample_rate"],
            manifest_filepath=cfg["manifest_filepath"],
            int_values=cfg.get("int_values", False),
            max_duration=cfg.get("max_duration", None),
            min_duration=cfg.get("min_duration", None),
            max_utts=cfg.get("max_utts", 0),
            ignore_index=cfg.get("blank_index", -1),
            unk_index=cfg.get("unk_index", -1),
            normalize=cfg.get("normalize_transcripts", False),
            trim=cfg.get("trim_silence", False),
            parser=cfg.get("parser", "en"),
            augmentor=augmentor,
        )

        dataset, num_samples = dataset_creator.create()

        return dataset, num_samples

    def train(self, num_replicas):
        callbacks = []

        if "model_checkpoint" in self.trainer_cfg:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    **self.trainer_cfg["model_checkpoint"]
                )
            )

        if "tensorboard" in self.trainer_cfg:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(**self.trainer_cfg["tensorboard"])
            )

        super(Transducer, self).fit(
            x=self.train_dl,
            steps_per_epoch=self.train_steps,
            validation_data=self.validation_dl,
            validation_steps=self.validation_steps,
            epochs=self.trainer_cfg["epochs"],
            workers=self.trainer_cfg["workers"],
            max_queue_size=self.trainer_cfg["max_queue_size"],
            use_multiprocessing=self.trainer_cfg["use_multiprocessing"],
            callbacks=callbacks,
        )

    def call(self, batch, training):
        audio_signals, audio_lens, transcripts, _ = batch

        audio_signals, audio_lens = self.preprocessor(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
        )

        if training:
            audio_signals = self.spec_augment(audio_signals=audio_signals)

        encoded_signals, encoded_lens, _, _ = self.encoder(
            audio_signals=audio_signals, audio_lens=audio_lens
        )
        del audio_signals, audio_lens

        decoded_targets = self.predictor(targets=transcripts)
        del transcripts

        joint_outputs = self.joint(
            encoder_outputs=encoded_signals, predictor_outputs=decoded_targets
        )

        return (
            encoded_signals,
            encoded_lens,
            decoded_targets,
            joint_outputs,
        )

    def train_step(self, batch):
        audio, _, transcripts, transcript_lens = batch

        with tf.GradientTape() as tape:
            (
                _,
                encoded_lens,
                _,
                joint_outputs,
            ) = self(batch, training=True)

            loss = self.loss(
                (joint_outputs, encoded_lens), (transcripts, transcript_lens)
            )

            if self.mixed_precision_enabled:
                loss = self.optimizer.get_scaled_loss(loss)

        # scaling and unscaling for amp
        gradients = tape.gradient(loss, self.trainable_weights)
        if self.mixed_precision_enabled:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # TODO: clip gradient
        # TODO: vairational noise

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # TODO: compute wer & cer for every n step
        # TODO: log wer & cer & loss & lr

        return {"train_loss": loss}

    def test_step(self, batch):
        _, _, transcripts, _ = batch

        (
            _,
            encoded_lens,
            _,
            decoded_lens,
            joint_outputs,
        ) = self(batch, training=False)

        loss = self.loss(
            log_probs=joint_outputs,
            targets=transcripts,
            encoded_lens=encoded_lens,
            decoded_lens=decoded_lens,
        )

        # TODO: compute wer & cer

        return {"val_loss": loss}
