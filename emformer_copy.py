import torch
import torch.nn as nn
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf

from modules.emformer_encoder import EmformerEncoder
from modules.greedy_inference import GreedyInference
from modules.transducer_decoder import TransducerDecoder
from modules.transducer_joint import TransducerJoint
from modules.transducer_predictor import TransducerPredictor

initialize(config_path="configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char.yaml")


class Emformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        self.labels = cfg.labels

        self.preprocessor = instantiate(cfg.preprocessor)
        
        encoder_args = OmegaConf.to_container(cfg.model.encoder)
        encoder_args.pop("_target_")
        self.encoder = torch.jit.script(EmformerEncoder(**encoder_args))

        predictor_args = OmegaConf.to_container(cfg.model.predictor)
        predictor_args.pop("_target_")
        self.predictor = torch.jit.script(TransducerPredictor(**predictor_args, vocab_size=len(self.labels)))

        joint_args = OmegaConf.to_container(cfg.model.joint)
        joint_args.pop("_target_")
        self.joint = torch.jit.script(
            TransducerJoint(
                **joint_args,
                encoder_hidden=cfg.model.encoder.dim_model,
                predictor_hidden=cfg.model.predictor.dim_model,
                vocab_size=len(self.labels)
            )
        )

        self.decoder = torch.jit.script(TransducerDecoder(labels=labels, inference=inference))

        inference_args = OmegaConf.to_container(cfg.model.inference)
        inference_args.pop("_target_")
        self.inference = torch.jit.script(
            GreedyInference(
                **inference_args,
                predictor=predictor,
                joint=joint,
                blank_index=len(self.labels),
            )
        )


    def forward(
        self,
        audio_signals,
        audio_lens,
        mode="full_context",
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        with torch.no_grad():
            cache_rnn_state, cache_k, cache_v = cache

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

            current_hypotheses, cache_rnn_state = self.decoder(
                encoded_signals,
                encoded_lens,
                cache_rnn_state,
                mode="stream",
            )

        return current_hypotheses, (cache_rnn_state, cache_k, cache_v)


if __name__ == "__main__":
    encoder_args = OmegaConf.to_container(cfg.model.encoder)
    encoder_args.pop("_target_")

    predictor_args = OmegaConf.to_container(cfg.model.predictor)
    predictor_args.pop("_target_")

    joint_args = OmegaConf.to_container(cfg.model.joint)
    joint_args.pop("_target_")

    preprocessor_args = OmegaConf.to_container(cfg.model.preprocessor)
    preprocessor_args.pop("_target_")

    encoder = torch.jit.script(EmformerEncoder(**encoder_args))

    # audio_signals = torch.randn(1, 80, 240).to(dtype=torch.float32)
    # audio_lens = torch.tensor([240])
    # cache_k = torch.randn(16, 1, 20, 8, 64).to(dtype=torch.float32)  # 2.6MB
    # cache_v = torch.randn(16, 1, 20, 8, 64).to(dtype=torch.float32)  # 2.6MB
    # mode = "stream"

    # output = encoder(audio_signals, audio_lens, cache_k, cache_v, mode)

    predictor = torch.jit.script(TransducerPredictor(**predictor_args, vocab_size=28))
    joint = torch.jit.script(
        TransducerJoint(
            **joint_args,
            encoder_hidden=cfg.model.encoder.dim_model,
            predictor_hidden=cfg.model.predictor.dim_model,
            vocab_size=28
        )
    )

    inference = torch.jit.script(
        GreedyInference(
            predictor=predictor,
            joint=joint,
            blank_index=29,
            max_symbols_per_step=30,
        )
    )

    decoder = torch.jit.script(TransducerDecoder(labels=labels, inference=inference))
