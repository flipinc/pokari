import tensorflow as tf
from frontends.audio_featurizer import AudioFeaturizer
from hydra.core import global_hydra
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from modules.inference import Inference
from modules.mock_stream import MockStream
from omegaconf import OmegaConf

tf.random.set_seed(2)


def transducer_setup(cfgs):
    audio_featurizer = AudioFeaturizer(**OmegaConf.to_container(cfgs.audio_feature))
    text_featurizer = instantiate(cfgs.text_feature)

    encoder = instantiate(cfgs.encoder)
    predictor = instantiate(cfgs.predictor, num_classes=29)
    joint = instantiate(cfgs.joint, num_classes=29)

    inference = Inference(
        text_featurizer=text_featurizer,
        predictor=predictor,
        joint=joint,
    )

    stream = MockStream(
        audio_featurizer, text_featurizer, encoder, predictor, inference
    )

    return stream


class TestMockStream:
    def test_mock_rnnt_stream(self):
        """test mock rnnt stream runs without any error"""

        global_hydra.GlobalHydra.instance().clear()

        initialize(config_path="../../configs/rnnt", job_name="rnnt")
        cfgs = compose(config_name="librispeech_char.yml")

        stream = transducer_setup(cfgs)

        audio_signals = tf.random.normal([24000])
        stream(audio_signals)

    def test_mock_emformer_stream(self):
        """test mock rnnt stream runs without any error"""

        global_hydra.GlobalHydra.instance().clear()

        initialize(config_path="../../configs/emformer", job_name="emformer")
        cfgs = compose(config_name="librispeech_char_mini_stack.yml")

        stream = transducer_setup(cfgs)

        audio_signals = tf.random.normal([24000])
        stream(audio_signals)
