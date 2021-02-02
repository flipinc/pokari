import hydra
import pytorch_lightning as pl
from models.transducer import Transducer


@hydra.main(config_name="configs/emformer/emformer_en.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    model = Transducer(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)


if __name__ == "__main__":
    main()
