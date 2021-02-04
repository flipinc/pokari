import json
import logging
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
from omegaconf import DictConfig, OmegaConf
from optimizers.lr_scheduler import compute_max_steps


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
            decoder=self.decoder,
            batch_dim_index=0,
            use_cer=False,
            log_prediction=True,
            dist_sync_on_step=True,
        )

        optim_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)

        self.lr_scheduler_config = (
            optim_cfg.pop("lr_scheduler") if "lr_scheduler" in optim_cfg else None
        )

        self.variatinoal_noise_config = (
            optim_cfg.pop("variational_noise")
            if "variational_noise" in optim_cfg
            else None
        )

        optimizer = optim_cfg.pop("name")
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), **optim_cfg)
        else:
            raise NotImplementedError(f"{optimizer} is not yet supported.")

    def configure_optimizers(self):
        if self.trainer.distributed_backend is None:
            num_workers = self.trainer.num_gpus or 1
        elif self.trainer.distributed_backend == "ddp_cpu":
            num_workers = self.trainer.num_processes * self.trainer.num_nodes
        elif self.trainer.distributed_backend == "ddp":
            num_workers = self.trainer.num_gpus * self.trainer.num_nodes
        else:
            logging.warning(
                "The lightning trainer received accelerator: "
                f"{self.trainer.distributed_backend}. We recommend to "
                "use 'ddp' instead."
            )
            num_workers = self.trainer.num_gpus * self.trainer.num_nodes

        max_steps = compute_max_steps(
            max_epochs=self.trainer.max_epochs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            limit_train_batches=self.trainer.limit_train_batches,
            num_workers=num_workers,
            num_samples=len(self.train_dl.dataset),
            batch_size=self.train_dl.batch_size,
            drop_last=self.train_dl.drop_last,
        )

        if self.lr_scheduler_config:
            warmup_ratio = self.lr_scheduler_config.pop("warmup_ratio")
            self.warmup_steps = max_steps * warmup_ratio
            self.lr_scheduler = instantiate(
                self.lr_scheduler_config,
                self.optimizer,
                max_steps=max_steps,
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
    def simulate(
        self, paths2audio_files: List[str], batch_size: int = 4, mode="normal"
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
        is_training = self.training
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
                    audio_signals = test_batch[0].to(device)
                    audio_lens = test_batch[1].to(device)

                    if mode == "normal":
                        audio_signals, audio_lens = self.preprocessor(
                            audio_signals=audio_signals,
                            audio_lens=audio_lens,
                        )

                        encoded_signals, encoded_lens = self.encoder(
                            audio_signals=audio_signals,
                            audio_lens=audio_lens,
                        )

                        hypotheses += self.decoder.generate_hypotheses(
                            encoded_signals, encoded_lens
                        )[0]
                    elif mode == "stream":
                        t = audio_signals.size(1)

                        if batch_size > 1:
                            raise ValueError(
                                "Only support batch size = 1 for streaming simulation "
                                "because all audio lengths must be aligned"
                            )

                        base_length = (
                            self.preprocessor.hop_length
                            * self.encoder.subsampling_factor
                        )

                        pre_chunk_length = base_length * self.encoder.chunk_length
                        pre_right_length = base_length * self.encoder.right_length

                        num_chunks = math.ceil(t / pre_chunk_length)

                        cache_rnn_state = None
                        cache_k = cache_v = None
                        for i in range(num_chunks):
                            start_offset = pre_chunk_length * i
                            end_offset = (
                                pre_chunk_length * (i + 1)
                                if pre_chunk_length * (i + 1) <= t
                                else t
                            )

                            remaining_space = t - (end_offset + pre_right_length)
                            if remaining_space >= 0:
                                this_right_length = pre_right_length
                            elif remaining_space < 0:
                                this_right_length = t - end_offset

                            end_offset = end_offset + this_right_length

                            (
                                current_hypotheses,
                                cache_k,
                                cache_v,
                                cache_rnn_state,
                            ) = self.emformer_stream(
                                audio_signals=audio_signals[:, start_offset:end_offset],
                                audio_lens=torch.Tensor([end_offset - start_offset]).to(
                                    device
                                ),
                                cache_rnn_state=cache_rnn_state,
                                cache_k=cache_k,
                                cache_v=cache_v,
                            )

                            hypotheses += current_hypotheses

                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=is_training)
        return hypotheses

    def emformer_stream(
        self,
        audio_signals,
        audio_lens=None,
        cache_rnn_state=None,
        cache_k=None,
        cache_v=None,
    ):
        if audio_lens is None:
            bs = audio_signals.size(0)
            # assuming all audio_signal has segment length (chunk length + right length)
            base_length = self.preprocessor.hop_length * self.encoder.subsampling_factor

            chunk_length = base_length * self.encoder.chunk_length
            right_length = base_length * self.encoder.right_length

            audio_lens = torch.Tensor([chunk_length + right_length] * bs).to(
                audio_signals.device
            )

        audio_signals, audio_lens = self.preprocessor(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
        )

        encoded_signals, encoded_lens, cache_k, cache_v = self.encoder.stream(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
            cache_k=cache_k,
            cache_v=cache_v,
        )

        current_hypotheses, cache_rnn_state = self.decoder.generate_hypotheses(
            encoded_signals,
            encoded_lens,
            cache_rnn_state,
            mode="stream",
        )

        return current_hypotheses, cache_k, cache_v, cache_rnn_state

    def setup_dataloader_from_config(self, cfg: DictConfig):
        augmentor = get_augmentations(cfg["augmentor"]) if "augmentor" in cfg else None
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

    def forward(
        self,
        audio_signals,
        # when streaming for the first iteration, audio_lens is unknown
        audio_lens=None,
        transcripts=None,
        transcript_lens=None,
        mode="normal",
        cache_rnn_state=None,
        # for emformer
        cache_k=None,
        cache_v=None,
    ):
        # TODO: Because ONNX only supports forward() for inference, streaming code is
        # forcefully included. This makes this code messy as the output is completely
        # different.
        if mode == "normal":
            audio_signals, audio_lens = self.preprocessor(
                audio_signals=audio_signals,
                audio_lens=audio_lens,
            )

            if self.training:
                audio_signals = self.spec_augment(audio_signals=audio_signals)

            encoded_signals, encoded_lens = self.encoder(
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
        elif mode == "stream":
            with torch.no_grad():
                is_training = self.training

                try:
                    self.eval()

                    (
                        current_hypotheses,
                        cache_k,
                        cache_v,
                        cache_rnn_state,
                    ) = self.emformer_stream(
                        audio_signals=audio_signals,
                        audio_lens=audio_lens,
                        cache_k=cache_k,
                        cache_v=cache_v,
                        cache_rnn_state=cache_rnn_state,
                    )

                finally:
                    self.train(mode=is_training)

            return current_hypotheses, audio_lens, cache_k, cache_v, cache_rnn_state

    def training_step(self, batch, batch_idx):
        audio_signals, audio_lens, transcripts, transcript_lens = batch

        (
            encoded_signals,
            encoded_lens,
            decoded_targets,
            decoded_lens,
            joint_outputs,
        ) = self.forward(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
            transcripts=transcripts,
            transcript_lens=transcript_lens,
        )
        del audio_signals, audio_lens

        loss_value = self.loss(
            log_probs=joint_outputs,
            targets=transcripts,
            encoded_lens=encoded_lens,
            decoded_lens=decoded_lens,
        )

        tensorboard_logs = {
            "train_loss": loss_value,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.wer.update(encoded_signals, encoded_lens, transcripts, transcript_lens)
            _, scores, words = self.wer.compute()
            tensorboard_logs.update({"training_batch_wer": scores.float() / words})

        self.log_dict(tensorboard_logs)

        return {"loss": loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signals, audio_lens, transcripts, transcript_lens = batch

        (
            encoded_signals,
            encoded_lens,
            decoded_targets,
            decoded_lens,
            joint_outputs,
        ) = self.forward(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
            transcripts=transcripts,
            transcript_lens=transcript_lens,
        )
        del audio_signals, audio_lens

        loss_value = self.loss(
            log_probs=joint_outputs,
            targets=transcripts,
            encoded_lens=encoded_lens,
            decoded_lens=decoded_lens,
        )

        tensorboard_logs = {"val_loss": loss_value}

        self.wer.update(encoded_signals, encoded_lens, transcripts, transcript_lens)
        wer, wer_num, wer_denom = self.wer.compute()

        tensorboard_logs["val_wer_num"] = wer_num
        tensorboard_logs["val_wer_denom"] = wer_denom
        tensorboard_logs["val_wer"] = wer

        self.log_dict(tensorboard_logs)

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
                for param_name, param in self.predictor.named_parameters():
                    if param.grad is not None:
                        noise = torch.normal(
                            mean=mean,
                            std=std,
                            size=param.size(),
                            device=param.device,
                            dtype=param.dtype,
                        )
                        param.grad.data.add_(noise)
