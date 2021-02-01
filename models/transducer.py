import json
import math
import os
import tempfile
from typing import Dict, List

import pytorch_lightning as pl
import torch
from frontends.audio_augment import AudioAugmentor, SpeedPerturbation
from frontends.audio_preprocess import AudioToMelSpectrogramPreprocessor
from frontends.audio_to_text import AudioToCharDataset
from frontends.spec_augment import SpectrogramAugmentation
from losses.transducer import RNNTLoss
from metrics.wer import RNNTWER

# from modules.emformer_encoder import EmformerEncoder
from modules.transducer_decoder import RNNTDecoding
from modules.transducer_joint import RNNTJoint
from modules.transducer_predictor import RNNTDecoder
from optimizers.lr_scheduler import CosineAnnealing


class EncDecRNNTModel(pl.LightningModule):
    """Base class for encoder decoder RNNT-based models."""

    def __init__(self, trainer: pl.Trainer = None):
        super().__init__()

        labels = [
            " ",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "'",
        ]
        self.labels = labels

        self._train_dl = None
        self._validation_dl = None
        self._trainer = trainer

        self.setup_training_data()
        self.setup_validation_data()

        ###
        # Encoder
        ###

        # for contextNet
        # enc_hidden = 1024

        # for conformer
        # enc_hidden = 256

        # for emformer
        enc_hidden = 512

        ###
        # Predictor
        ###

        # for contextNet, conformer
        pred_hidden = 320
        pred_num_layers = 1
        emded_dim = 320

        # for emformer
        # pred_hidden = 512
        # pred_num_layers = 2
        # emded_dim = 256

        ###
        # Joint
        ###

        # for contextNet, conformer
        joint_hidden = 320

        # for emformer
        # joint_hidden = 640

        # Initialize components
        self.preprocessor = AudioToMelSpectrogramPreprocessor()
        self.spec_augmentation = SpectrogramAugmentation()

        # self.encoder = ConvASREncoder(conv_asr_encoder)
        # self.encoder = ConformerEncoder(conformer_asr_encoder)
        # self.encoder = EmformerEncoder(emformer_encoder_config)
        self.decoder = RNNTDecoder(
            pred_hidden=pred_hidden,
            num_layers=pred_num_layers,
            vocab_size=len(labels),
            emded_dim=emded_dim,
        )
        self.joint = RNNTJoint(
            enc_hidden=enc_hidden,
            pred_hidden=pred_hidden,
            joint_hidden=joint_hidden,
            num_classes=len(labels),
        )

        self.loss = RNNTLoss(num_classes=len(labels))

        # Setup decoding objects
        self.decoding = RNNTDecoding(
            decoder=self.decoder, joint=self.joint, vocabulary=labels
        )
        # Setup WER calculation
        self.wer = RNNTWER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=False,
            log_prediction=True,
            dist_sync_on_step=True,
        )

        self._optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.0005,
            betas=(0.9, 0.999),
            # for contextNet (probably this was the cause of degradation)
            # weight_decay=0.001
            # for conformer
            weight_decay=0.0001,
        )
        self._scheduler = CosineAnnealing(
            self._optimizer,
            max_steps=400000,
            min_lr=1e-6,
            warmup_steps=20000,  # 5% of total steps
        )

    def configure_optimizers(self):
        return ([self._optimizer], [{"scheduler": self._scheduler, "interval": "step"}])

    def train_dataloader(self):
        if self._train_dl is not None:
            return self._train_dl

    def val_dataloader(self):
        if self._validation_dl is not None:
            return self._validation_dl

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
                    "paths2audio_files": paths2audio_files,
                    "batch_size": batch_size,
                    "temp_dir": tmpdir,
                }

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in temporary_datalayer:
                    if streaming is False:
                        encoded, encoded_len = self.forward(
                            input_signal=test_batch[0].to(device),
                            input_signal_length=test_batch[1].to(device),
                        )

                        hypotheses += self.decoding.rnnt_decoder_predictions_tensor(
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
                            ) = self.decoding.rnnt_decoder_predictions_tensor(
                                encoded, encoded_len.to(device), hidden, streaming=True,
                            )
                            hypotheses += current_hypotheses

                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return hypotheses

    def _setup_dataloader_from_config(sel, training, batch_size=None, **kwargs):
        dataset = AudioToCharDataset(**kwargs)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,  # if batch_size else BATCH_SIZE,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            shuffle=training,
            num_workers=4,
            pin_memory=True,
        )

    def setup_training_data(self):
        augmentation = SpeedPerturbation(sr=16000, resample_type="kaiser_fast")

        self._train_dl = self._setup_dataloader_from_config(
            training=True,
            manifest_filepath="drive/MyDrive/jynote/manifest_train.json",
            labels=self.labels,
            sample_rate=16000,
            int_values=False,
            max_duration=None,
            min_duration=None,
            max_utts=0,
            blank_index=-1,
            unk_index=-1,
            normalize=True,
            trim=False,
            load_audio=True,
            parser="en",
            add_misc=False,
            augmentor=AudioAugmentor([(1.0, augmentation)]),
        )

    def setup_validation_data(self):
        self._validation_dl = self._setup_dataloader_from_config(
            training=False,
            manifest_filepath="drive/MyDrive/jynote/manifest_val.json",
            labels=self.labels,
            sample_rate=16000,
            int_values=False,
            max_duration=None,
            min_duration=None,
            max_utts=0,
            blank_index=-1,
            unk_index=-1,
            normalize=True,
            trim=False,
            load_audio=True,
            parser="en",
            add_misc=False,
        )

    def forward(self, input_signal=None, input_signal_length=None):
        processed_signal, processed_signal_length = self.preprocessor(
            x=input_signal, seq_len=input_signal_length,
        )

        if self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal)

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        return encoded, encoded_len

    def training_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch

        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len
        )
        del signal

        decoder, target_length = self.decoder(
            targets=transcript, target_length=transcript_len
        )

        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss_value = self.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )

        tensorboard_logs = {
            "train_loss": loss_value,
            "learning_rate": self._optimizer.param_groups[0]["lr"],
        }

        if (self._trainer.global_step + 1) % self._trainer.log_every_n_steps == 0:
            self.wer.update(encoded, encoded_len, transcript, transcript_len)
            _, scores, words = self.wer.compute()
            tensorboard_logs.update({"training_batch_wer": scores.float() / words})

        self.log_dict(tensorboard_logs)

        return {"loss": loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len
        )
        del signal

        tensorboard_logs = {}

        decoder, target_length = self.decoder(
            targets=transcript, target_length=transcript_len
        )
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

        loss_value = self.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )

        tensorboard_logs["val_loss"] = loss_value

        self.wer.update(encoded, encoded_len, transcript, transcript_len)
        wer, wer_num, wer_denom = self.wer.compute()

        tensorboard_logs["val_wer_num"] = wer_num
        tensorboard_logs["val_wer_denom"] = wer_denom
        tensorboard_logs["val_wer"] = wer

        return tensorboard_logs

    def _setup_transcribe_dataloader(
        self, config: Dict
    ) -> "torch.utils.data.DataLoader":
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be
                relatively short fragments. Recommended length per file is between 5
                and 25 seconds.
            batch_size: (int) batch size to use during inference. Bigger will result
                in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is
                temporarily stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        temporary_datalayer = self._setup_dataloader_from_config(
            training=False,
            manifest_filepath=os.path.join(config["temp_dir"], "manifest.json"),
            sample_rate=self.preprocessor.sample_rate,
            labels=self.labels,
            batch_size=min(config["batch_size"], len(config["paths2audio_files"])),
            trim=True,
        )
        return temporary_datalayer

    def on_after_backward(self):
        super().on_after_backward()
        # TODO: add varidational noise
