import json
import math
import os
import tempfile
from typing import Dict, List

import pytorch_lightning as pl
import torch
from datasets.audio_to_text import AudioToCharDataset
from frontends.audio_augment import get_augmentations
from hydra.utils import instantiate
from losses.transducer import TransducerLoss
from metrics.wer import TransducerWER
from modules.transducer_decoder import TransducerDecoder
from omegaconf import DictConfig


class Transducer(pl.LightningModule):
    """Lightning Module for Transducer models."""

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer):
        super().__init__()

        self.labels = cfg.labels

        self.train_dl = self.setup_dataloader_from_config(cfg.train_ds)
        self.validation_dl = self.setup_dataloader_from_config(cfg.validation_ds)
        self.trainer = trainer

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

        self.loss = TransducerLoss(vocab_size=len(self.labels))
        self.decoder = TransducerDecoder(
            predictor=self.predictor, joint=self.joint, labels=self.labels
        )
        self.wer = TransducerWER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=False,
            log_prediction=True,
            dist_sync_on_step=True,
        )

        self.lr_scheduler_config = (
            cfg.optimizer.pop("lr_scheduler")
            if "lr_scheduler" in cfg.optimizer
            else None
        )

        self.variatinoal_noise_config = (
            cfg.optimizer.pop("variational_noise")
            if "variational_noise" in cfg.optimizer
            else None
        )

        optimizer = cfg.optimizer.pop("name")
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), **cfg.optimizer)
        else:
            raise NotImplementedError(f"{optimizer} is not yet supported.")

    def setup(self, stage):
        if stage == "fit":
            total_devices = self.hparams.n_gpus * self.hparams.n_nodes
            train_batches = len(self.train_dataloader()) // total_devices
            self.total_train_steps = (
                self.hparams.epochs * train_batches
            ) // self.hparams.accumulate_grad_batches

    def configure_optimizers(self):
        if self.lr_scheduler_config:
            warmup_ratio = self.lr_scheduler_config.pop("warmup_ratio")
            self.warmup_steps = self.total_train_steps * warmup_ratio
            self.lr_scheduler = instantiate(
                self.lr_scheduler_config,
                self.optimizer,
                max_steps=self.total_train_steps,
                warmup_steps=self.warmup_steps,
            )
            return (
                [self.optimizer],
                [{"scheduler": self.lr_scheduler, "interval": "step"}],
            )
        else:
            self.warmup_steps = 0
            return self.optimizer

    def train_dataloader(self):
        if self.train_dl is not None:
            return self.train_dl

    def val_dataloader(self):
        if self.validation_dl is not None:
            return self.validation_dl

    @torch.no_grad()
    def transcribe(
        self, paths2audio_files: List[str], batch_size: int = 4, streaming=False
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging
        and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. Recommended length per
                file is between 5 and 25 seconds. But it is possible to pass a few hours
                long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. Bigger will result in
                better throughput performance but would use more memory.

        Returns:

            A list of transcriptions in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}
        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        try:
            # Switch model to evaluation mode
            self.eval()
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "manifest.json"), "w") as fp:
                    for audio_file in paths2audio_files:
                        entry = {
                            "audio_filepath": audio_file,
                            "duration": 100000,
                            "text": "nothing",
                        }
                        fp.write(json.dumps(entry) + "\n")

                config = {
                    "audio_filepaths": paths2audio_files,
                    "batch_size": batch_size,
                    "temp_dir": tmpdir,
                }

                temporary_datalayer = self.setup_transcribe_dataloader(config)
                for test_batch in temporary_datalayer:
                    if streaming is False:
                        encoded, encoded_len = self.forward(
                            input_signal=test_batch[0].to(device),
                            input_signal_length=test_batch[1].to(device),
                        )

                        hypotheses += self.decoding.generate_hypotheses(
                            encoded, encoded_len
                        )[0]
                    else:
                        processed_signal, processed_signal_length = self.preprocessor(
                            x=test_batch[0].to(device),
                            seq_len=test_batch[1].to(device),
                        )

                        chunk_length = 128  # 10ms per frame * 128 = 1280ms
                        right_length = 32  # 10ms per frame * 32 = 320ms

                        t = processed_signal.size(2)
                        num_chunks = math.ceil(t / chunk_length)

                        cache_q = cache_v = cache_audio = None
                        hidden = None
                        for i in range(num_chunks):
                            start_offset = chunk_length * i
                            end_offset = (
                                chunk_length * (i + 1)
                                if chunk_length * (i + 1) <= t
                                else t
                            )

                            remaining_space = t - (end_offset + right_length)
                            if remaining_space >= 0:
                                this_right_length = right_length
                            elif remaining_space < 0:
                                this_right_length = t - end_offset

                            end_offset = end_offset + this_right_length
                            end_offset_wo_padding = (
                                end_offset
                                if end_offset <= processed_signal_length.int()
                                else processed_signal_length.int()
                            )

                            (
                                encoded,
                                encoded_len,
                                cache_q,
                                cache_v,
                                cache_audio,
                            ) = self.encoder.recognize(
                                audio_signal=processed_signal[
                                    :, :, start_offset:end_offset
                                ],
                                length=torch.Tensor(
                                    [end_offset_wo_padding - start_offset]
                                ),
                                cache_q=cache_q,
                                cache_v=cache_v,
                                cache_audio=cache_audio,
                            )

                            # Make decoder stateful
                            (
                                current_hypotheses,
                                hidden,
                            ) = self.decoding.generate_hypotheses(
                                encoded,
                                encoded_len.to(device),
                                hidden,
                                streaming=True,
                            )
                            hypotheses += current_hypotheses

                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return hypotheses

    def setup_dataloader_from_config(self, cfg: DictConfig):
        augmentor = get_augmentations(cfg["augmentor"])
        dataset = AudioToCharDataset(
            manifest_filepath=cfg["manifest_filepath"],
            labels=self.labels,
            sample_rate=cfg["sample_rate"],
            int_values=cfg.get("int_values", False),
            augmentor=augmentor,
            max_duration=cfg.get("max_duration", None),
            min_duration=cfg.get("min_duration", None),
            max_utts=cfg.get("max_utts", 0),
            blank_index=cfg.get("blank_index", -1),
            unk_index=cfg.get("unk_index", -1),
            normalize=cfg.get("normalize_transcripts", False),
            trim=cfg.get("trim_silence", False),
            load_audio=cfg.get("load_audio", True),
            parser=cfg.get("parser", "en"),
            add_misc=cfg.get("add_misc", False),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get("drop_last", False),
            shuffle=cfg["shuffle"],
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
        )

    def forward(self, audio_signals=None, audio_lens=None):
        audio_signals, audio_lens = self.preprocessor(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
        )

        if self.training:
            audio_signals = self.spec_augmentation(audio_signals=audio_signals)

        encoded_signals, encoded_lens = self.encoder(
            audio_signals=audio_signals, audio_lens=audio_lens
        )

        return encoded_signals, encoded_lens

    def training_step(self, batch, batch_idx):
        audio_signals, audio_lens, transcripts, transcript_lens = batch

        encoded_signals, encoded_lens = self.forward(
            audio_signal=audio_signals, audio_lens=audio_lens
        )
        del audio_signals, audio_lens

        decoded_targets, decoded_lens = self.predictor(
            targets=transcripts, target_lens=transcript_lens
        )

        joint_outputs = self.joint(
            encoder_outputs=encoded_signals, predictor_ouputs=decoded_targets
        )
        loss_value = self.loss(
            log_probs=joint_outputs,
            targets=transcripts,
            encoded_lens=encoded_lens,
            decoded_lens=decoded_lens,
        )

        tensorboard_logs = {
            "train_loss": loss_value,
            "learning_rate": self._optimizer.param_groups[0]["lr"],
        }

        if (self._trainer.global_step + 1) % self._trainer.log_every_n_steps == 0:
            self.wer.update(encoded_signals, encoded_lens, transcripts, transcript_lens)
            _, scores, words = self.wer.compute()
            tensorboard_logs.update({"training_batch_wer": scores.float() / words})

        self.log_dict(tensorboard_logs)

        return {"loss": loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signals, audio_lens, transcripts, transcript_lens = batch

        encoded_signals, encoded_lens = self.forward(
            audio_signals=audio_signals, audio_lens=audio_lens
        )
        del audio_signals, audio_lens

        tensorboard_logs = {}

        decoded_transcripts, decoded_lens = self.decoder(
            transcripts=transcripts, transcript_lens=transcript_lens
        )
        joint_outputs = self.joint(
            encoded_signals=encoded_signals, decoded_transcripts=decoded_transcripts
        )

        loss_value = self.loss(
            log_probs=joint_outputs,
            transcripts=transcripts,
            encoded_lens=encoded_lens,
            decoded_lens=decoded_lens,
        )

        tensorboard_logs["val_loss"] = loss_value

        self.wer.update(encoded_signals, encoded_lens, transcripts, transcript_lens)
        wer, wer_num, wer_denom = self.wer.compute()

        tensorboard_logs["val_wer_num"] = wer_num
        tensorboard_logs["val_wer_denom"] = wer_denom
        tensorboard_logs["val_wer"] = wer

        return tensorboard_logs

    def setup_transcribe_dataloader(self, config: Dict):
        dl_config = {
            "manifest_filepath": os.path.join(config["temp_dir"], "manifest.json"),
            "sample_rate": self.preprocessor.sample_rate,
            "labels": self.labels,
            "batch_size": min(config["batch_size"], len(config["audio_filepaths"])),
            "trim_silence": True,
        }

        temporary_datalayer = self.setup_dataloader_from_config(DictConfig(dl_config))

        return temporary_datalayer

    def on_after_backward(self):
        super().on_after_backward()

        if self.variatinoal_noise_config:
            mean = self.variatinoal_noise_config["mean"]
            std = self.variatinoal_noise_config["std"]
            start_step = (
                self.warmup_steps
                if self.variatinoal_noise_config["start_step"] == -1
                else self.variatinoal_noise_config["start_step"]
            )

            if std > 0 and self.global_step >= start_step:
                for param_name, param in self.decoder.named_parameters():
                    if param.grad is not None:
                        noise = torch.normal(
                            mean=mean,
                            std=std,
                            size=param.size(),
                            device=param.device,
                            dtype=param.dtype,
                        )
                        param.grad.data.add_(noise)
