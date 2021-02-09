import pytorch_lightning as pl
import torch
from hydra.experimental import compose, initialize

from models.transducer import Transducer

initialize(config_path="configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char.yaml")


def main():
    trainer = pl.Trainer(**cfg.trainer)
    model = Transducer(cfg=cfg.model, trainer=trainer, return_torchscript=True).cuda()
    # model = Transducer.load_from_checkpoint(
    #     "../datasets/epoch=3-step=28539.ckpt",
    #     cfg=cfg.model,
    #     trainer=trainer,
    # ).cuda()

    print(
        model.simulate(
            ["../datasets/train/train-clean-100" "/1034/121119/1034-121119-0022.flac"],
            1,
            "stream",
        )
    )
    print(
        model.simulate(
            ["../datasets/train/train-clean-100" "/1034/121119/1034-121119-0022.flac"],
            1,
        )
    )

    audio_signals = torch.randn(2, 25600).cuda()
    audio_lens = torch.Tensor([25600, 25600]).cuda()
    cache_rnn_state = torch.randn(2, 1, 2, 320).cuda()
    cache_k = torch.randn(16, 2, 20, 8, 64).cuda()  # 2.6MB
    cache_v = torch.randn(16, 2, 20, 8, 64).cuda()  # 2.6MB
    cache = (cache_rnn_state, cache_k, cache_v)

    out = model(
        audio_signals=audio_signals,
        audio_lens=audio_lens,
        mode="stream",
        cache=cache,
    )

    torch.onnx.export(
        model,
        (
            audio_signals,
            audio_lens,
            "",
            "",
            "stream",
            cache,
        ),
        "emformer.onnx",
        input_names=["audio_signals", "audio_lens", "cache"],
        output_names=["hypothesis", "cache"],
        example_outputs=out,
        opset_version=12,
    )


if __name__ == "__main__":
    main()
