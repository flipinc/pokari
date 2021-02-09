from typing import Tuple

import torch
import torch.nn as nn
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf

from frontends.audio_preprocess import AudioToMelSpectrogramPreprocessor
from modules.emformer_encoder import EmformerEncoder
from modules.greedy_inference import GreedyInference
from modules.transducer_decoder import TransducerDecoder
from modules.transducer_joint import TransducerJoint
from modules.transducer_predictor import TransducerPredictor


class Emformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.labels = cfg.labels

        preprocessor_args = OmegaConf.to_container(cfg.preprocessor)
        preprocessor_args.pop("_target_")
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preprocessor_args)

        encoder_args = OmegaConf.to_container(cfg.encoder)
        encoder_args.pop("_target_")
        self.encoder = EmformerEncoder(**encoder_args)

        predictor_args = OmegaConf.to_container(cfg.predictor)
        predictor_args.pop("_target_")
        self.predictor = TransducerPredictor(
            **predictor_args, vocab_size=len(self.labels)
        )

        joint_args = OmegaConf.to_container(cfg.joint)
        joint_args.pop("_target_")
        self.joint = TransducerJoint(
            **joint_args,
            encoder_hidden=cfg.encoder.dim_model,
            predictor_hidden=cfg.predictor.dim_model,
            vocab_size=len(self.labels),
        )

        inference_args = OmegaConf.to_container(cfg.inference)
        inference_args.pop("_target_")
        self.inference = GreedyInference(
            **inference_args,
            predictor=self.predictor,
            joint=self.joint,
            blank_index=len(self.labels),
        )

        self.decoder = TransducerDecoder(labels=self.labels, inference=self.inference)

    def forward(
        self,
        audio_signals,
        audio_lens,
        cache_rnn_state: Tuple[torch.Tensor, torch.Tensor],
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        with torch.no_grad():
            audio_signals, audio_lens = self.preprocessor(
                audio_signals=audio_signals,
                audio_lens=audio_lens,
            )

        encoded_signals, encoded_lens, cache_k, cache_v = self.encoder(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
            cache_k=cache_k,
            cache_v=cache_v,
            mode="stream",
        )

        hypotheses, cache_rnn_state = self.decoder(
            encoded_signals,
            encoded_lens,
            cache_rnn_state,
            mode="stream",
        )

        return hypotheses, cache_rnn_state, cache_k, cache_v


initialize(config_path="configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char.yaml")

if __name__ == "__main__":
    emformer = Emformer(cfg.model).cuda()

    audio_signals = torch.randn(2, 25600).cuda()
    audio_lens = torch.Tensor([25600, 25600]).cuda()
    cache_rnn_state = (torch.randn(1, 2, 320).cuda(), torch.randn(1, 2, 320).cuda())
    cache_k = torch.randn(16, 2, 20, 8, 64).cuda()  # 2.6MB
    cache_v = torch.randn(16, 2, 20, 8, 64).cuda()  # 2.6MB

    out = emformer(
        audio_signals=audio_signals,
        audio_lens=audio_lens,
        cache_rnn_state=cache_rnn_state,
        cache_k=cache_k,
        cache_v=cache_v,
    )

    torch.onnx.export(
        emformer,
        (
            audio_signals,
            audio_lens,
            cache_rnn_state,
            cache_k,
            cache_v,
        ),
        "emformer.onnx",
        example_outputs=out,
        opset_version=11,
    )
