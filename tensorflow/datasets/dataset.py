import os

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

BUFFER_SIZE = 100
TFRECORD_SHARDS = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(
        self,
        stage: str,
        audio_featurizer,
        text_featurizer,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        num_print_sample_data: int = 0,
        **kwargs,
    ):
        self.audio_featurizer = audio_featurizer
        self.text_featurizer = text_featurizer
        self.data_paths = data_paths
        self.cache = cache  # whether to cache WHOLE transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size  # shuffle buffer size
        self.stage = stage  # for defining tfrecords files
        self.drop_remainder = (
            drop_remainder  # whether to drop remainder for multi gpu training
        )
        self.steps_per_epoch = None  # for better training visualization

        self.num_print_sample_data = num_print_sample_data

    def create(self, batch_size: int):
        self.read_entries()
        if not self.steps_per_epoch or self.steps_per_epoch == 0:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)

    def read_entries(self):
        self.entries = []

        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                self.entries += f.read().splitlines()

        self.entries = [line.split("\t", 1) for line in self.entries]

        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join(
                [str(x) for x in self.text_featurizer.extract(line[-1]).numpy()]
            )

        self.entries = np.array(self.entries)

        if self.shuffle:
            np.random.shuffle(self.entries)  # Mix transcripts.tsv

        self.steps_per_epoch = len(self.entries)

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes):
            wave, rate = librosa.load(
                os.path.expanduser(path.decode("utf-8")), sr=None, mono=True
            )
            wave = tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)
            return wave.numpy()

        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)

        return record[0], audio, record[1]

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        if self.num_print_sample_data:
            for d, _ in dataset.take(self.num_print_sample_data):
                print("\n🎤 audio_signals:\n", d["audio_signals"])
                print("\n🔭 audio_lens:\n", d["audio_lens"])
                print("\n🎯 targets:\n", d["targets"])
                print("\n👀 target_lens:\n", d["target_lens"])

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                {
                    "audio_signals": tf.TensorShape([None]),
                    "audio_lens": tf.TensorShape([]),
                    "targets": tf.TensorShape([None]),
                    "target_lens": tf.TensorShape([]),
                },
                {"labels": tf.TensorShape([None]), "label_lens": tf.TensorShape([])},
            ),
            padding_values=(
                {
                    "audio_signals": 0.0,
                    "audio_lens": 0,
                    "targets": self.text_featurizer.blank,
                    "target_lens": 0,
                },
                {"labels": self.text_featurizer.blank, "label_lens": 0},
            ),
            drop_remainder=self.drop_remainder,
        )

        dataset = dataset.prefetch(AUTOTUNE)

        # TODO: is this the right way?

        # if self.drop_remainder:
        #     self.steps_per_epoch = math.floor(float(len(dataset)) / float(batch_size))
        # else:
        #     self.steps_per_epoch = math.ceil(float(len(dataset)) / float(batch_size))

        self.steps_per_epoch = len(dataset)

        return dataset

    @tf.function
    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        def tf_read_raw_audio(audio: tf.Tensor, sample_rate=16000):
            wave, rate = tf.audio.decode_wav(
                audio, desired_channels=1, desired_samples=-1
            )
            resampled = tfio.audio.resample(
                wave, rate_in=tf.cast(rate, dtype=tf.int64), rate_out=sample_rate
            )
            return tf.reshape(resampled, shape=[-1])  # reshape for using tf.signal

        with tf.device("/CPU:0"):
            audio_signal = tf_read_raw_audio(audio, self.audio_featurizer.sample_rate)

            audio_len = tf.cast(tf.shape(audio_signal)[0], tf.int32)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_len = tf.cast(tf.shape(label)[0], tf.int32)

            target = self.text_featurizer.prepand_blank(label)
            target_len = tf.cast(tf.shape(target)[0], tf.int32)

        return (
            {
                "audio_signals": audio_signal,
                "audio_lens": audio_len,
                "targets": target,
                "target_lens": target_len,
            },
            {"labels": label, "label_lens": label_len},
        )
