import torch
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from modules.greedy_inference import GreedyInference
from modules.transducer_joint import TransducerJoint
from modules.transducer_predictor import TransducerPredictor

initialize(config_path="../configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char_pytorch.yml")

if __name__ == "__main__":
    cfg_predictor = OmegaConf.to_container(cfg.model.predictor)
    cfg_predictor.pop("_target_")
    predictor = TransducerPredictor(**cfg_predictor, vocab_size=29)

    cfg_joint = OmegaConf.to_container(cfg.model.joint)
    cfg_joint.pop("_target_")
    joint = TransducerJoint(
        **cfg_joint, vocab_size=29, encoder_hidden=512, predictor_hidden=320
    )

    encoded_outs = torch.randn(4, 512, 60)
    encoded_lens = torch.Tensor([60, 60, 60, 60])

    inference = GreedyInference(
        predictor, joint, blank_index=0, max_symbols_per_step=30
    )

    inference(encoded_outs, encoded_lens)
