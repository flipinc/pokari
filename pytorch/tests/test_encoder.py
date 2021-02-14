import torch
from hydra.utils import instantiate
from modules.emformer_encoder import EmformerEncoder
from omegaconf import OmegaConf

subsampling_factor = 4

cfg = OmegaConf.create(
    {
        "_target_": "modules.emformer_encoder.EmformerEncoder",
        "subsampling": "vgg",
        "subsampling_factor": subsampling_factor,
        "subsampling_dim": 256,
        "feat_in": 80,
        "num_layers": 16,
        "num_heads": 8,
        "dim_model": 512,
        "dim_ffn": 2048,
        "dropout_attn": 0.1,
        "left_length": 20,
        "chunk_length": 32,
        "right_length": 8,
    }
)


class TestEmformer:
    def test_constructor(self):
        model = instantiate(cfg)
        assert isinstance(model, EmformerEncoder)

    def test_mask(self):
        model = instantiate(cfg, left_length=2, chunk_length=3, right_length=2)

        t_max = 10  # padded length
        input_len = 7  # actual length

        audio_lens = torch.Tensor([input_len])
        mask, right_indexes = model.create_mask(
            audio_lens=audio_lens, t_max=t_max, device=audio_lens.device
        )

        # 1. number of copied right indexes should match.
        # num_chunks(3) * right_length(2) - trim_last_space(1) = 5
        # this includes padded timesteps
        assert len(right_indexes) == 5

        # 2. size should match
        assert mask.size(0) == mask.size(1) == 1
        assert mask.size(2) == mask.size(3) == 10 + len(right_indexes)

        # 3. Padding is taken into account for all masks
        # False(0) in mask means should attention is calculated

        # mask_diagnal
        assert mask[0, 0, 0, 0] == 0
        assert mask[0, 0, 0, 0] == mask[0, 0, 1, 1] == mask[0, 0, 2, 2]
        assert mask[0, 0, 3, 3] == 1
        assert mask[0, 0, 3, 3] == mask[0, 0, 4, 4]

        # mask_left (mask_right is just a transposed version of mask_left)
        assert torch.sum(mask[0, 0, 5:8, 0:2] == 0) == 6
        assert torch.sum(mask[0, 0, 8:11, 2:4] == 0) == 3
        assert torch.sum(mask[0, 0, 11:14, 4:5] == 0) == 0

        # mask_body
        assert torch.sum(mask[0, 0, 5:8, 5:8] == 0) == 9  # without left context
        assert torch.sum(mask[0, 0, 8:11, 6:11] == 0) == 15  # with left context
        assert torch.sum(mask[0, 0, 11:14, 9:14] == 0) == 3  # padding
