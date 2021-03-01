import tensorflow as tf
from frontends.audio_featurizer import AudioFeaturizer


class TestFrontend:
    def test_audio_len(self):
        audio_featurizer = AudioFeaturizer()

        audio_signals = tf.random.normal((1, 24000))
        audio_lens = tf.constant([24000], tf.int32)

        _, target_audio_lens = audio_featurizer(audio_signals, audio_lens)

        source_audio_lens = tf.cast(
            tf.math.ceil(audio_lens / audio_featurizer.hop_length),
            tf.int32,
        )

        assert tf.equal(target_audio_lens, source_audio_lens)
