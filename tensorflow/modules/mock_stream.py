import tensorflow as tf

from modules.emformer_encoder import EmformerEncoder
from modules.rnnt_encoder import RNNTEncoder


class MockStream:
    def __init__(
        self, audio_featurizer, text_featurizer, encoder, predictor, inference
    ):
        self.audio_featurizer = audio_featurizer
        self.encoder = encoder
        self.predictor = predictor
        self.inference = inference

        if isinstance(encoder, EmformerEncoder):
            self.fn = self.emformer_stream
        elif isinstance(encoder, RNNTEncoder):
            self.fn = self.rnnt_stream
        else:
            NotImplementedError("Mock streaming is not implemented for this encoder.")

    def __call__(self, audio_signals, **kwargs):
        """

        Args:
            audio_signals: Audio signal without batch dimension [T]

        """
        with tf.device("/CPU:0"):
            if len(tf.shape(audio_signals)) > 1:
                raise ValueError(
                    "Only support batch size = 1 for streaming simulation "
                    "because all audio lengths must be aligned"
                )

            return self.fn(audio_signals, **kwargs)

    def emformer_stream(self, audio_signals, **kwargs):
        base_length = self.audio_featurizer.hop_length * self.encoder.subsampling_factor

        # actual length (chunk/right length is calculated relative to what it receives)
        chunk_length = base_length * self.encoder.chunk_length
        right_length = base_length * self.encoder.right_length

        t = tf.shape(audio_signals)[0]
        num_chunks = tf.math.ceil(t / chunk_length)

        prev_encoder_cache = self.encoder.get_initial_state(batch_size=1)
        prev_predictor_cache = self.predictor.get_initial_state(batch_size=1)
        prev_index = self.text_featurizer.blank * tf.ones(shape=[], dtype=tf.int32)
        for i in tf.range(num_chunks):
            start_offset = chunk_length * i
            end_offset = chunk_length * (i + 1) if chunk_length * (i + 1) <= t else t

            remaining_space = t - (end_offset + right_length)
            if remaining_space >= 0:
                this_right_length = right_length
            elif remaining_space < 0:
                this_right_length = t - end_offset

            end_offset = end_offset + this_right_length

            chunk_audio_signal = audio_signals[start_offset:end_offset]
            chunk_audio_signal = tf.expand_dims(
                chunk_audio_signal, axis=0
            )  # add batch dim
            chunk_audio_lens = tf.constant([end_offset - start_offset])

            audio_features, _ = self.audio_featurizer.tf_extract(
                chunk_audio_signal, chunk_audio_lens
            )

            encoded_outs, prev_encoder_cache = self.encoder.stream(
                audio_features, prev_encoder_cache
            )
            encoded_outs = tf.squeeze(encoded_outs, axis=0)  # remove batch dim

            hypothesis = self.inference.greedy_decode(
                encoded_outs,
                tf.shape(encoded_outs)[0],
                prev_index,
                prev_predictor_cache,
            )

            prev_index = hypothesis.index
            prev_predictor_cache = hypothesis.states

            prediction = self.text_featurizer.indices2upoints(hypothesis.prediction)

            tf.print("🎨: ", i, prediction)

    def rnnt_stream(self, audio_signals, features_per_stream: int = 30):
        """

        For computational efficiency, multiple frames are bundled and fed to encoder.

        Args:
            features_per_stream: number of features to be fed into encoder. Better to
                be a multiple of reduction factors.

        """
        chunk_length = features_per_stream * self.audio_featurizer.hop_length

        t = tf.shape(audio_signals)[0]
        num_chunks = tf.math.ceil(t / chunk_length)

        prev_encoder_cache = self.encoder.get_initial_state(batch_size=1)
        prev_predictor_cache = self.predictor.get_initial_state(batch_size=1)
        prev_index = self.text_featurizer.blank * tf.ones(shape=[], dtype=tf.int32)
        for i in tf.range(num_chunks):
            start_offset = chunk_length * i
            end_offset = chunk_length * (i + 1) if chunk_length * (i + 1) <= t else t

            chunk_audio_signal = audio_signals[start_offset:end_offset]
            chunk_audio_signal = tf.expand_dims(
                chunk_audio_signal, axis=0
            )  # add batch dim
            chunk_audio_lens = tf.constant([end_offset - start_offset])

            audio_features, _ = self.audio_featurizer.tf_extract(
                chunk_audio_signal, chunk_audio_lens
            )

            encoded_outs, prev_encoder_cache = self.encoder.stream(
                audio_features, prev_encoder_cache
            )
            encoded_outs = tf.squeeze(encoded_outs, axis=0)  # remove batch dim

            hypothesis = self.inference.greedy_decode(
                encoded_outs,
                tf.shape(encoded_outs)[0],
                prev_index,
                prev_predictor_cache,
            )

            prev_index = hypothesis.index
            prev_predictor_cache = hypothesis.states

            prediction = self.text_featurizer.indices2upoints(hypothesis.prediction)

            tf.print("🎨: ", i, prediction)
