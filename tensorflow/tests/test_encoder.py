import numpy as np
import tensorflow as tf
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
        "num_layers": 16,
        "num_heads": 8,
        "dim_model": 512,
        "dim_ffn": 2048,
        "dropout_attn": 0.1,
        "dropout_ffn": 0.1,
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
        """

        should look like following:

        [[[ 3 [1 1 0 0 0 1 1 1 0 0 0 0 0 0 0]
            4 [1 1 0 0 0 1 1 1 0 0 0 0 0 0 0]
            6 [0 0 1 0 0 0 1 1 1 1 1 0 0 0 0]
            7 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            9 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            0 [1 1 0 0 0 1 1 1 0 0 0 0 0 0 0]
            1 [1 1 0 0 0 1 1 1 0 0 0 0 0 0 0]
            2 [1 1 0 0 0 1 1 1 0 0 0 0 0 0 0]
            3 [0 0 1 0 0 0 1 1 1 1 1 0 0 0 0]
            4 [0 0 1 0 0 0 1 1 1 1 1 0 0 0 0]
            5 [0 0 1 0 0 0 1 1 1 1 1 0 0 0 0]
            6 [0 0 0 0 0 0 0 0 0 1 1 1 0 0 0]
            7 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            8 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            9 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ]]]
               3 4 6 7 9 0 1 2 3 4 5 6 7 8 9

        """
        model = instantiate(cfg, left_length=2, chunk_length=3, right_length=2)

        t_max = 10  # padded length
        input_len = 7  # actual length

        audio_lens = tf.constant([input_len])

        mask, right_indexes = model.np_create_mask(
            audio_lens=audio_lens.numpy(), t_max=t_max
        )

        # 1. number of copied right indexes should match.
        # num_chunks(3) * right_length(2) - trim_last_space(1) = 5
        # this includes padded timesteps
        assert len(right_indexes) == 5

        # 2. size should match
        assert mask.shape[0] == mask.shape[1] == 1
        assert mask.shape[2] == mask.shape[3] == 10 + len(right_indexes)

        # 3. Padding is taken into account for all masks
        # False(0) in mask means should attention is calculated

        # mask_diagnal
        assert np.sum(mask[0, 0, 0:2, 0:2]) == 4
        assert np.sum(mask[0, 0, 0:5, 0:5]) == 5

        # mask_left
        assert np.sum(mask[0, 0, 5:8, 0:2]) == 6
        assert np.sum(mask[0, 0, 8:11, 2:4]) == 3
        assert np.sum(mask[0, 0, 11:14, 4:5]) == 0

        # mask_right
        assert np.sum(mask[0, 0, 0:2, 5:8]) == 6
        assert np.sum(mask[0, 0, 2:4, 6:11]) == 5

        # mask_body
        # without left context
        assert np.sum(mask[0, 0, 5:8, 5:8]) == 9
        # with left context
        assert np.sum(mask[0, 0, 8:11, 6:11]) == 15
        # padding
        assert np.sum(mask[0, 0, 11:14, 9:14]) == 3

    def test_full_context(self):
        model = instantiate(cfg)

        audio_features = tf.random.normal((4, 480, 80))
        audio_lens = tf.constant([480, 400, 380, 420], tf.int32)

        x, audio_lens = model(audio_features, audio_lens)

        assert isinstance(x, tf.Tensor)
        assert isinstance(audio_lens, tf.Tensor)


class TestConformer:
    def test_mask(self):
        """test padding mask is valid"""
        assert 1 == 1


class TestJasper:
    def test_mask(self):
        """test padding mask is valid"""
        assert 1 == 1
