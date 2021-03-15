import pytorch_lightning as pl
from hydra.experimental import compose, initialize

from models.transducer import Transducer

initialize(config_path="../configs/emformer", job_name="emformer")
cfg = compose(config_name="librispeech_char_pytorch.yml")


def main():
    trainer = pl.Trainer(**cfg.trainer)
    model = Transducer(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)


if __name__ == "__main__":
    main()
