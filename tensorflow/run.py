# TODO: this is temporary file

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

import tensorflow as tf
from datasets.audio_to_text import DatasetCreator
from frontends.audio_preprocess import AudioToMelSpectrogramPreprocessor
from frontends.spec_augment import SpectrogramAugmentation
from modules.emformer_encoder import EmformerEncoder

initialize(config_path="../configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char_tensorflow.yml")

if __name__ == "__main__":
    ################
    # dataset
    ################

    # creator = DatasetCreator(
    #     batch_size=4,
    #     stage="val",
    #     tfrecords_dir="../datasets",
    #     manifest_filepath="../datasets/manifest_val.json",
    #     labels=[
    #         " ",
    #         "a",
    #         "b",
    #         "c",
    #         "d",
    #         "e",
    #         "f",
    #         "g",
    #         "h",
    #         "i",
    #         "j",
    #         "k",
    #         "l",
    #         "m",
    #         "n",
    #         "o",
    #         "p",
    #         "q",
    #         "r",
    #         "s",
    #         "t",
    #         "u",
    #         "v",
    #         "w",
    #         "x",
    #         "y",
    #         "z",
    #         "'",
    #     ],
    #     sample_rate=16000,
    # )

    # datasets = creator.create()

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
    cfg = OmegaConf.to_container(cfg.model.encoder)
    cfg.pop("_target_")
    # cfg.pop("subsampling")
    # emformer = EmformerEncoder(**cfg, subsampling="stack")

    emformer = EmformerEncoder(**cfg)

    audio_signals = tf.random.normal([4, 80, 360])
    audio_lens = tf.constant([360, 360, 360, 360])

    emformer(audio_signals, audio_lens, mode="stream")
