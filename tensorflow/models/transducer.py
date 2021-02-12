import tensorflow as tf
from audio_augment import get_augmentations
from datasets.audio_to_text import DatasetCreator
from hydra.utils import instantiate
from omegaconf import DictConfig


class Transducer(tf.keras.Model):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.labels = cfg.labels

        self.train_dl = self.setup_dataloader_from_config(cfg.train_ds)
        self.validation_dl = self.setup_dataloader_from_config(cfg.validation_ds)

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

        self.inference = instantiate(
            cfg.inference,
            predictor=self.predictor,
            joint=self.joint,
            blank_index=len(self.labels),
        )

        self.decoder = TransducerDecoder(labels=self.labels, inference=self.inference)

        self.stream = instantiate(
            cfg.stream,
            preprocessor=self.preprocessor,
            encoder=self.encoder,
            decoder=self.decoder,
        )

        self.wer = TransducerWER(
            decoder=self.decoder,
            batch_dim_index=0,
            use_cer=False,
            log_prediction=True,
            dist_sync_on_step=True,
        )

        loss = TransducerLoss(vocab_size=len(self.labels))
        optimizer = tf.keras.optimizers.get(optimizer)
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            optimizer, "dynamic"
        )

        self.compile(optimizer=optimizer, loss=loss)

    def setup_dataloader_from_config(self, cfg: DictConfig):
        augmentor = get_augmentations(cfg["augmentor"]) if "augmentor" in cfg else None
        dataset = DatasetCreator(
            batch_size=cfg["batch_size"],
            tfrecords_dir=cfg["tfrecords_dir"],
            stage=cfg["stage"],
            cache=cfg["cache"],
            shuffle=cfg["shuffle"],
            tfrecords_shards=cfg["tfrecords_shards"],
            manifest_filepath=cfg["manifest_filepath"],
            labels=self.labels,
            sample_rate=cfg["sample_rate"],
            int_values=cfg.get("int_values", False),
            drop_remainder=cfg.get("drop_remainder", False),
            augmentor=augmentor,
            max_duration=cfg.get("max_duration", None),
            min_duration=cfg.get("min_duration", None),
            max_utts=cfg.get("max_utts", 0),
            ignore_index=cfg.get("blank_index", -1),
            unk_index=cfg.get("unk_index", -1),
            normalize=cfg.get("normalize_transcripts", False),
            trim=cfg.get("trim_silence", False),
            load_audio=cfg.get("load_audio", True),
            parser=cfg.get("parser", "en"),
        )
        return dataset.create()

    def train(self, cfg: DictConfig):
        callbacks = [
            # tf.keras.callbacks.ModelCheckpoint
            # tf.keras.callbacks.experimental.BackupAndRestore
            # tf.keras.callbacks.TensorBoard
        ]

        super(Transducer, self).fit(
            x=self.train_dl,
            validation_data=self.validation_dl,
            epochs=cfg.epochs,
            workers=cfg.workers,
            steps_per_epoch=cfg.steps_per_epoch,
            callbacks=callbacks,
        )

    def call(self, batch, training=False):
        audio_signals, audio_lens, transcripts, transcript_lens = batch

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

        decoded_targets, decoded_lens = self.predictor(
            targets=transcripts, target_lens=transcript_lens
        )
        del transcripts, transcript_lens

        joint_outputs = self.joint(
            encoder_outputs=encoded_signals, predictor_outputs=decoded_targets
        )

        return (
            encoded_signals,
            encoded_lens,
            decoded_targets,
            decoded_lens,
            joint_outputs,
        )

    @tf.function
    def train_step(self, batch):
        _, _, transcripts, _ = batch

        with tf.GradientTape() as tape:
            (
                _,
                encoded_lens,
                _,
                decoded_lens,
                joint_outputs,
            ) = self(batch)

            loss = self.loss(
                log_probs=joint_outputs,
                targets=transcripts,
                encoded_lens=encoded_lens,
                decoded_lens=decoded_lens,
            )

            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # scaling and unscaling for amp
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # TODO: compute wer & cer for every n step

        return {"train_loss": loss, "learning_rate": None}

    @tf.function
    def test_step(self, batch):
        _, _, transcripts, _ = batch

        (
            _,
            encoded_lens,
            _,
            decoded_lens,
            joint_outputs,
        ) = self(batch)

        loss = self.loss(
            log_probs=joint_outputs,
            targets=transcripts,
            encoded_lens=encoded_lens,
            decoded_lens=decoded_lens,
        )

        # TODO: compute wer & cer

        return {"val_loss": loss}
