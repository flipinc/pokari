from frontends.audio_featurizer import AudioFeaturizer
from frontends.spec_augment import SpectrogramAugmentation
from hydra.utils import instantiate
from losses.ctc_loss import CTCLoss
from omegaconf import DictConfig, OmegaConf

from models.base_model import BaseModel


class CTC(BaseModel):
    def __init__(
        self,
        cfgs: DictConfig,
        global_batch_size: int,
        setup_training: bool = True,
    ):
        audio_featurizer = AudioFeaturizer(**OmegaConf.to_container(cfgs.audio_feature))
        text_featurizer = instantiate(cfgs.text_feature)

        super().__init__(
            cfgs=cfgs,
            global_batch_size=global_batch_size,
            setup_training=setup_training,
            audio_featurizer=audio_featurizer,
            text_featurizer=text_featurizer,
            loss_module=CTCLoss,
        )

        self.spec_augment = SpectrogramAugmentation(
            **OmegaConf.to_container(cfgs.spec_augment)
        )

        self.encoder = instantiate(cfgs.encoder)
        self.decoder = instantiate(
            cfgs.decoder, num_classes=self.text_featurizer.num_classes
        )

    def call(self, inputs, training=False):
        audio_signals = inputs["audio_signals"]
        audio_lens = inputs["audio_lens"]

        # [B, T, n_mels]
        audio_features, audio_lens = self.audio_featurizer(audio_signals, audio_lens)

        # [B, T, n_mels]
        if training:
            audio_features = self.spec_augment(audio_features)

        # [B, T, D_e]
        encoded_outs, encoded_lens = self.encoder(audio_features, audio_lens)

        # [B, T, num_classes]
        logits = self.decoder(encoded_outs=encoded_outs)

        return {"logits": logits, "logit_lens": encoded_lens}
