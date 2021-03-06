import tensorflow as tf
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from modules.inference import Inference

initialize(config_path="../../configs/rnnt", job_name="rnnt")
cfgs = compose(config_name="librispeech_char.yml")

tf.random.set_seed(2)


class MockTextFeaturizer:
    def __init__(self):
        self.blank = 0

    def iextract(self, labels):
        return labels


text_featurizer = MockTextFeaturizer()
predictor = instantiate(cfgs.predictor, num_classes=29)
joint = instantiate(cfgs.joint, num_classes=29)
inference = Inference(
    batch_size=3, text_featurizer=text_featurizer, predictor=predictor, joint=joint
)


class TestInference:
    def test_batch_once(self):
        """test if two batch decode outputs the same result"""

        encoded_outs = tf.random.normal((3, 240, 512))  # [B, T, D]
        encoded_lens = tf.constant([220, 240, 230])

        pred_1, _, _ = inference.greedy_batch_decode(
            encoded_outs=encoded_outs, encoded_lens=encoded_lens, max_symbols=1
        )
        pred_2, _, _ = inference.greedy_batch_decode_once(
            encoded_outs=encoded_outs, encoded_lens=encoded_lens
        )

        assert tf.reduce_sum(pred_1[0]) == tf.reduce_sum(pred_2[0])
        assert tf.reduce_sum(pred_1[1]) == tf.reduce_sum(pred_2[1])
        assert tf.reduce_sum(pred_1[2]) == tf.reduce_sum(pred_2[2])

    def test_batch_decode(self):
        """test if two batch decoders return the same result"""

        encoded_outs = tf.random.normal((3, 240, 512))  # [B, T, D]
        encoded_lens = tf.constant([220, 240, 230])

        pred_1, _, _ = inference.greedy_batch_decode(
            encoded_outs=encoded_outs, encoded_lens=encoded_lens
        )
        result_1 = text_featurizer.iextract(pred_1)

        pred_2 = inference.greedy_naive_batch_decode(
            encoded_outs=encoded_outs, encoded_lens=encoded_lens
        )
        result_2 = text_featurizer.iextract(pred_2)

        # have to use reduce_sum because both handle blank symbol in a different way
        assert tf.reduce_sum(result_1[0]) == tf.reduce_sum(result_2[0])
        assert tf.reduce_sum(result_1[1]) == tf.reduce_sum(result_2[1])
        assert tf.reduce_sum(result_1[2]) == tf.reduce_sum(result_2[2])

    def test_cache(self):
        """test if given cache, return the same result

        Note: naive batch decode does not have caching mechanism so, instead use
        naive decode to compare with batch decode with batch size = 1

        """

        encoded_outs = tf.random.normal((1, 32, 512))  # [1, T, D_e]
        encoded_lens = tf.constant([32])  # [1]
        prev_tokens = tf.expand_dims(tf.constant([12]), axis=-1)  # [1, 1]
        cache_states = tf.random.normal(
            (cfgs.predictor.num_layers, 2, 1, cfgs.predictor.dim_model)
        )  # [N, 2, 1, D_p]

        _, next_token_1, next_states_1 = inference.greedy_batch_decode(
            encoded_outs=encoded_outs,
            encoded_lens=encoded_lens,
            prev_tokens=prev_tokens,
            cache_states=cache_states,
        )
        hypothesis = inference.greedy_decode(
            encoded_out=encoded_outs[0],
            encoded_len=encoded_lens[0],
            prev_token=prev_tokens[0],
            cache_states=cache_states,
        )
        next_token_2 = hypothesis.index
        next_states_2 = hypothesis.states

        assert tf.reduce_sum(next_token_1) == tf.reduce_sum(next_token_2)
        assert tf.reduce_sum(next_states_1) == tf.reduce_sum(next_states_2)
