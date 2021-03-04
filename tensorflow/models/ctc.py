import tensorflow as tf
from hydra.utils import instantiate
from losses.ctc_loss import CTCLoss
from modules.ctc_decoder import CTCDecoder
from omegaconf import DictConfig

from models.base_model import BaseModel


class CTC(BaseModel):
    def __init__(
        self,
        cfgs: DictConfig,
        global_batch_size: int,
        setup_training: bool = True,
    ):
        super(CTC, self).__init__(
            loss_module=CTCLoss,
            cfgs=cfgs,
            global_batch_size=global_batch_size,
            setup_training=setup_training,
        )

        self.encoder = instantiate(cfgs.encoder)
        self.decoder = CTCDecoder(num_classes=self.text_featurizer.num_classes)
        self.summarize_lists = [self.encoder, self.decoder]

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
        logits = self.decoder(encoded_outs)

        return {
            "logits": logits,
            "logit_lens": encoded_lens,
            "encoded_outs": encoded_outs,
        }

    # TODO: this is definitely not the right way to use input signature
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None, None], dtype=tf.float32),
            tf.TensorSpec([None], dtype=tf.int32),
            tf.TensorSpec([None, None, None], dtype=tf.float32),
        ]
    )
    def on_step_end(self, labels, logits, logit_lens, encoded_outs):
        if self.log_interval is not None and tf.equal(
            tf.math.floormod(self.step_counter + 1, self.log_interval), 0
        ):
            # get argmax of first item in batch
            log_prob = tf.math.argmax(logits[0], axis=-1)

            hypothesis = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

            blank = tf.constant(self.text_featurizer.blank, tf.int32)
            previous = tf.constant(self.text_featurizer.blank, tf.int32)
            for char_idx in log_prob:
                char_idx = tf.cast(char_idx, tf.int32)

                if (
                    tf.not_equal(char_idx, previous) or tf.equal(previous, blank)
                ) and tf.not_equal(char_idx, blank):
                    hypothesis = hypothesis.write(hypothesis.size(), char_idx)

                previous = char_idx

            hypothesis = tf.expand_dims(hypothesis.stack(), axis=0)  # add batch dim
            prediction = self.text_featurizer.iextract(hypothesis)

            tf.print("❓ PRED: \n", prediction[0])
            tf.print(
                "🧩 TRUE: \n",
                tf.strings.unicode_encode(
                    self.text_featurizer.indices2upoints(labels[0]), "UTF-8"
                ),
            )

            # hypotheses = []
            # # Drop predictions to CPU
            # prediction_cpu_tensor = log_probs.long().cpu()
            # # iterate over batch
            # for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            #     prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            #     # CTC decoding procedure
            #     decoded_prediction = []
            #     previous = self.blank_id
            #     for p in prediction:
            #         if (
            #             p != previous or previous == self.blank_id
            #         ) and p != self.blank_id:
            #             decoded_prediction.append(p)
            #         previous = p
            #     hypothesis = "".join([self.labels_map[c] for c in decoded_prediction])
            #     hypotheses.append(hypothesis)
