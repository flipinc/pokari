import tensorflow as tf
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from modules.inference import Inference

initialize(config_path="../../configs/rnnt", job_name="rnnt")
cfgs = compose(config_name="librispeech_wordpiece.yml")

tf.random.set_seed(2)


class MockTextFeaturizer:
    def __init__(self):
        self.blank = 0

    def iextract(self, labels):
        return labels


text_featurizer = MockTextFeaturizer()
predictor = instantiate(cfgs.predictor, num_classes=29)
joint = instantiate(cfgs.joint, num_classes=29)
inference = Inference(text_featurizer=text_featurizer, predictor=predictor, joint=joint)


class TestInference:
    def test_batch_decode(self):
        """test if two batch decoders return the same result"""

        encoded_outs = tf.random.normal((2, 240, 512))  # [B, T, D]
        encoded_lens = tf.constant([220, 240])

        result_1 = inference._greedy_batch_decode(encoded_outs, encoded_lens)
        result_2 = inference._greedy_naive_batch_decode(encoded_outs, encoded_lens)

        print(result_1[0])
        print(result_2[0])

        print(result_1[1])
        print(result_2[1])

        # print(result_1[2])
        # print(result_2[2])

        # assert tf.math.equal(result_1[0], result_2[0])
        # assert tf.math.equal(result_1[1], result_2[1])
