# TODO: this is temporary file

import numpy as np
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

import tensorflow as tf
from datasets.audio_to_text import DatasetCreator
from frontends.audio_preprocess import AudioToMelSpectrogramPreprocessor
from frontends.spec_augment import SpectrogramAugmentation
from losses.transducer import TransducerLoss
from modules.emformer_encoder import EmformerEncoder
from modules.transducer_joint import TransducerJoint
from modules.transducer_predictor import TransducerPredictor

initialize(config_path="../configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char_tensorflow.yml")

if __name__ == "__main__":
    ################
    # dataset
    ################

    creator = DatasetCreator(
        batch_size=4,
        stage="val",
        tfrecords_dir="../datasets",
        manifest_filepath="../datasets/manifest_val.json",
        labels=[
            " ",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "'",
        ],
        sample_rate=16000,
    )

    datasets = creator.create()

    print(next(iter(datasets)))

    ################
    # preprocessor
    ################

    # preprocessor = AudioToMelSpectrogramPreprocessor()

    # audio = tf.random.normal([4, 10000])
    # audio_lens = tf.constant([10000, 10000, 10000, 10000])

    # preprocessor(audio, audio_lens)

    ################
    # specaug
    ################

    # spec_augment = SpectrogramAugmentation()

    # audio = tf.random.normal([4, 80, 120])
    # spec_augment(audio)

    ################
    # emformer
    ################
    # cfg_encoder = OmegaConf.to_container(cfg.model.encoder)
    # cfg_encoder.pop("_target_")

    # stack mode
    # cfg.pop("subsampling")
    # emformer = EmformerEncoder(**cfg, subsampling="stack")

    # vgg mode
    # encoder = EmformerEncoder(**cfg_encoder)

    # audio_signals = tf.random.normal([4, 80, 360])
    # audio_lens = tf.constant([360, 360, 360, 360])

    # encoder(audio_signals, audio_lens, mode="stream")

    ################
    # predictor
    ################
    # cfg_predictor = OmegaConf.to_container(cfg.model.predictor)
    # cfg_predictor.pop("_target_")
    # predictor = TransducerPredictor(**cfg_predictor, vocab_size=29)

    # dummy = tf.range(1, 30, 2)
    # dummy = tf.expand_dims(dummy, axis=0)
    # targets = tf.tile(dummy, [4, 1])
    # target_lens = tf.constant([29, 29, 29, 29])

    # predictor(targets, target_lens)

    ################
    # joint
    ################
    # cfg_joint = OmegaConf.to_container(cfg.model.joint)
    # cfg_joint.pop("_target_")
    # joint = TrasducerJoint(
    #     **cfg_joint, vocab_size=29, encoder_hidden=512, predictor_hidden=320
    # )

    # dummy_encoder = tf.random.normal((4, 512, 240))
    # dummy_predictor = tf.random.normal((4, 320, 64))

    # joint(dummy_encoder, dummy_predictor)

    ################
    # loss
    ################

    # loss = TransducerLoss(vocab_size=29)

    # @tf.function
    # def run():
    #     audio_signals = tf.random.normal([4, 12000])
    #     audio_lens = np.array([12000, 12000, 12000, 12000])

    #     transcripts = tf.expand_dims(tf.range(1, 16), axis=0)
    #     transcripts = tf.tile(transcripts, [4, 1])
    #     transcript_lens = np.array([15, 15, 15, 15])

    #     audio_signals, audio_lens = preprocessor(
    #         audio_signals=audio_signals,
    #         audio_lens=audio_lens,
    #     )

    #     audio_signals = spec_augment(audio_signals=audio_signals)

    #     encoded_signals, encoded_lens, _, _ = encoder(
    #         audio_signals=audio_signals, audio_lens=audio_lens
    #     )
    #     del audio_signals, audio_lens

    #     decoded_targets = predictor(targets=transcripts)

    #     joint_outputs = joint(
    #         encoder_outputs=encoded_signals, predictor_outputs=decoded_targets
    #     )
    #     del encoded_signals, decoded_targets

    #     loss_value = loss(
    #         log_probs=joint_outputs,
    #         targets=transcripts,
    #         encoded_lens=encoded_lens,
    #         decoded_lens=transcript_lens,
    #     )

    # run()
