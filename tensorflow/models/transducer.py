import os

import librosa
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from losses.transducer_loss import TransducerLoss
from modules.inference import Inference
from modules.mock_stream import MockStream
from omegaconf import DictConfig

from models.base_model import BaseModel


class Transducer(BaseModel):
    """

    Some training tips about transducer model:
    - RNN encoder may end up with lower loss than transducer but higher CER/WER
    ref: https://github.com/espnet/espnet/issues/2606
    - For transducer models and CTC modesl, the first phase of learning is to learn all
    blanks.
    - The balance between predictor and encoder is super important.
    - Memory requirement is so high for transducer models. There are a couple of things
    you can try to decrease memory capacity
        1. reduce audio length (for transformer, be careful with the amount of contexts)
        2. reduce batch size
        3. reduce vocab size

    """

    def __init__(
        self,
        cfgs: DictConfig,
        global_batch_size: int,
        setup_training: bool = True,
    ):
        super(Transducer, self).__init__(
            loss_module=TransducerLoss,
            cfgs=cfgs,
            global_batch_size=global_batch_size,
            setup_training=setup_training,
        )

        self.encoder = instantiate(cfgs.encoder)
        self.predictor = instantiate(
            cfgs.predictor, num_classes=self.text_featurizer.num_classes
        )
        self.joint = instantiate(
            cfgs.joint, num_classes=self.text_featurizer.num_classes
        )
        self.summarize_lists = [self.encoder, self.predictor, self.joint]

        self.inference = Inference(
            batch_size=global_batch_size,
            text_featurizer=self.text_featurizer,
            predictor=self.predictor,
            joint=self.joint,
        )
        self.mock_stream = MockStream(
            audio_featurizer=self.audio_featurizer,
            text_featurizer=self.text_featurizer,
            encoder=self.encoder,
            predictor=self.predictor,
            inference=self.inference,
        )

    def call(self, inputs, training=False):
        audio_signals = inputs["audio_signals"]
        audio_lens = inputs["audio_lens"]
        targets = inputs["targets"]
        target_lens = inputs["target_lens"]

        # [B, T, n_mels]
        audio_features, audio_lens = self.audio_featurizer(audio_signals, audio_lens)

        # [B, T, n_mels]
        if training:
            audio_features = self.spec_augment(audio_features)

        # [B, T, D_e]
        encoded_outs, encoded_lens = self.encoder(audio_features, audio_lens)

        # [B, U, D_p]
        decoded_outs = self.predictor(targets, target_lens)

        # [B, T, U, D_j]
        logits = self.joint([encoded_outs, decoded_outs])

        return {
            "logits": logits,
            "logit_lens": encoded_lens,
            "encoded_outs": encoded_outs,
        }

    # TODO: this is definitely not the right way to use input signature
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None, None, None], dtype=tf.float32),
            tf.TensorSpec([None], dtype=tf.int32),
            tf.TensorSpec([None, None, None], dtype=tf.float32),
        ]
    )
    def on_step_end(self, labels, logits, logit_lens, encoded_outs):
        """

        TODO: Currently, because logging is not done at every step and graph mode
        requires func's output to be consistent, unnecessary logging is performed

        """
        # autograph does not allow intermediate return so seed initial values
        wer = self.wer.value
        cer = self.cer.value

        if self.log_interval is not None and tf.equal(
            tf.math.floormod(self.step_counter + 1, self.log_interval), 0
        ):
            preds, _, _ = self.inference.greedy_batch_decode(encoded_outs, logit_lens)
            preds = self.text_featurizer.indices2String(preds)
            labels = self.text_featurizer.indices2String(labels)

            tf.print("❓ PRED: \n", preds[0])
            tf.print("🧩 TRUE: \n", labels[0])

            wer = self.wer(preds, labels)
            cer = self.cer(preds, labels)

            tf.print("📕 WER: \n", self.wer(preds, labels))
            tf.print("📘 CER: \n", self.cer(preds, labels))

        return {"wer": wer, "cer": cer}

    def stream(
        self,
        manifest_idx: int = 0,
        enable_graph: bool = False,
    ):
        """Mock streaming on CPU by segmenting an audio into chunks

        Args:
            manifest_idx: Audio path and transcript pair to use for mock streaming.
            enable_graph: Enable graph mode and hide intermediate print outputs

        """
        path, transcript = self.train_ds.entries[manifest_idx]
        tf.print(f"🎙 Using {path}...")

        audio_signal, native_rate = librosa.load(
            os.path.expanduser(path),
            sr=self.audio_featurizer.sample_rate,
            mono=True,
            dtype=np.float32,
        )
        audio_signal = tf.convert_to_tensor(audio_signal)
        transcript = tf.strings.unicode_encode(
            self.text_featurizer.indices2upoints(
                tf.strings.to_number(tf.strings.split(transcript), out_type=tf.int32)
            ),
            output_encoding="UTF-8",
        )

        if enable_graph:
            self.fn = tf.function(self.mock_stream)
        else:
            self.fn = self.mock_stream

        self.fn(audio_signal)

        tf.print("💎: ", transcript)

    def stream_one_tflite(
        self, audio_signal, prev_token, cache_encoder_states, cache_predictor_states
    ):
        """Streaming tflite transducer model for batch size = 1
        Args:
            audio_signal: [T]
            prev_token: []
            cache_encoder_states: size depends on encoder type
            cache_predictor_states: [N, 2, B, D_p]
        Returns:
            tf.Tensor: transcript with size [None]
            tf.Tensor: last predicted token with size []
            tf.Tensor: new encoder states. size depends on encoder type
            tf.Tensor: new predictor states with size [N, 2, B, D_p]
        """
        audio_signal = tf.expand_dims(audio_signal, axis=0)  # add batch dim
        audio_len = tf.expand_dims(tf.shape(audio_signal)[1], axis=0)
        audio_feature = self.audio_featurizer.stream(audio_signal, audio_len)

        encoded_out, cache_encoder_states = self.encoder.stream(
            audio_feature, cache_encoder_states
        )
        encoded_out = tf.squeeze(encoded_out, axis=0)  # remove batch dim

        hypothesis = self.inference.greedy_decode(
            encoded_out=encoded_out,
            encoded_len=tf.shape(encoded_out)[0],
            prev_token=prev_token,
            cache_states=cache_predictor_states,
        )

        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)

        # if prediction is all blanks, it will cause an error because tflite expects
        # `transcript` to have a shape. If you see `ValueError: Invalid tensor size.`,
        # thats the error message. To prevent it, a fake symbol (or this could be used
        # as end-of-speech symbol) must be added.
        blank = tf.constant([0], tf.int32)
        transcript = tf.concat([blank, transcript], axis=0)

        return transcript, hypothesis.index, cache_encoder_states, hypothesis.states

    def stream_batch_tflite(
        self,
        audio_signals,
        prev_tokens,
        cache_encoder_states,
        cache_predictor_states,
    ):
        """Streaming tflite transducer model for batch size = n
        Args:
            audio_signal: [B, T]
            prev_token: [B, 1]
            cache_encoder_states: size depends on encoder type
            cache_predictor_states: [N, 2, B, D_p]
        Returns:
            tf.Tensor: transcript with size [B, None]
            tf.Tensor: last predicted token with size [B, 1]
            tf.Tensor: new encoder states. size depends on encoder type
            tf.Tensor: new predictor states with size [N, 2, B, D_p]
        """
        audio_lens = tf.expand_dims(tf.shape(audio_signals)[1], axis=0)
        audio_features = self.audio_featurizer.stream(audio_signals, audio_lens)

        encoded_outs, cache_encoder_states = self.encoder.stream(
            audio_features, cache_encoder_states
        )

        (
            predictions,
            prev_tokens,
            cache_predictor_states,
        ) = self.inference.greedy_batch_decode_once(
            encoded_outs=encoded_outs,  # [B, T, D_e]
            encoded_lens=tf.repeat(  # [B]
                tf.shape(encoded_outs)[1], [tf.shape(encoded_outs)[0]]
            ),
            prev_tokens=prev_tokens,  # [B]
            cache_states=cache_predictor_states,  # size depends on encoder type
        )

        transcripts = tf.TensorArray(
            tf.int32,
            size=tf.shape(predictions)[0],  # bs
            element_shape=tf.TensorShape([None]),  # max_length
        )
        ct = tf.constant(0, tf.int32)
        max_length = tf.shape(predictions)[1]
        for pred in predictions:
            transcript = self.text_featurizer.indices2upoints(pred)  # [None]
            transcript = tf.pad(transcript, [[0, max_length - tf.shape(transcript)[0]]])
            transcripts = transcripts.write(ct, transcript)
            ct += 1
        transcripts = transcripts.stack()

        return transcripts, prev_tokens, cache_encoder_states, cache_predictor_states

    def make_one_tflite_function(self):
        return tf.function(
            self.stream_one_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(
                    self.encoder.get_initial_state(batch_size=1).get_shape(),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    self.predictor.get_initial_state(batch_size=1).get_shape(),
                    dtype=tf.float32,
                ),
            ],
        )

    def make_batch_tflite_function(self, batch_size: int):
        return tf.function(
            self.stream_batch_tflite,
            input_signature=[
                tf.TensorSpec([batch_size, None], dtype=tf.float32),
                tf.TensorSpec([batch_size, 1], dtype=tf.int32),
                tf.TensorSpec(
                    self.encoder.get_initial_state(batch_size=batch_size).get_shape(),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    self.predictor.get_initial_state(batch_size=batch_size).get_shape(),
                    dtype=tf.float32,
                ),
            ],
        )
