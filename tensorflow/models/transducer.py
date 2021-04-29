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
        4. set `dynamic_memory_devices` so tensorflow does not use up all memory cap

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
        # Currently, because logging is not done at every step and graph mode
        # requires func's output to be consistent (autograph does not allow intermediate
        # return so seed initial values), unnecessary logging is performed
        wer = self.wer_value
        cer = self.cer_value

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

            # update metrics
            self.wer_value.assign(wer)
            self.cer_value.assign(cer)

            tf.print("📕 WER: \n", wer)
            tf.print("📘 CER: \n", cer)

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
        path, transcript, duration, offset = self.train_ds.entries[manifest_idx]
        tf.print(f"🎙 Using {path}...")

        offset = float(offset)
        duration = float(duration)
        # align with librosa format
        duration = None if duration == -1 else duration

        audio_signal, native_rate = librosa.load(
            path,
            sr=self.audio_featurizer.sample_rate,
            mono=True,
            duration=duration,
            offset=offset,
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
