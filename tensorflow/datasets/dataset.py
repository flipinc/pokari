import os

import librosa
import numpy as np
import tensorflow as tf

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
        buffer_size: int = 100,
        num_print_sample_data: int = 0,
        **kwargs,
    ):
        """

        Args:
            stage (str): the name of this dataset. currently not used
            cache (bool): cache WHOLE transformed dataset to memory
            shuffle (bool): shuffle tf.data.Dataset
            buffer_size (int): shuffle buffer size
            drop_remainder (bool): drop remainder for multi gpu training

        """
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")

        self.audio_featurizer = audio_featurizer
        self.text_featurizer = text_featurizer
        self.data_paths = data_paths
        self.cache = cache
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.drop_remainder = drop_remainder

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
        entries = []

        for file_path in self.data_paths:
            print(f"📘 Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                lines = f.read().splitlines()
                for data in lines:
                    # path, trascript, duration, offset
                    data = data.split("\t")

                    data[1] = " ".join(
                        [
                            str(x)
                            for x in self.text_featurizer.string2Indices(
                                data[1]
                            ).numpy()
                        ]
                    )

                    entries.append(data)

        self.entries = np.array(entries)
        self.steps_per_epoch = len(entries)

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes, duration: bytes, offset: bytes):
            offset = float(offset.decode())
            duration = float(duration.decode())
            # align with librosa format
            duration = None if duration == -1 else duration

            wave, rate = librosa.load(
                os.path.expanduser(path.decode("utf-8")),
                sr=16000,
                mono=True,
                duration=duration,
                offset=offset,
            )
            wave = tf.expand_dims(wave, axis=-1)  # add channel dim
            wave = tf.audio.encode_wav(wave, sample_rate=rate)

            return wave.numpy()

        # iterating over tf.Tensor is not allowed -> _, _, = list()
        path = record[0]
        transcript = record[1]
        duration = record[2]
        offset = record[3]

        audio = tf.numpy_function(fn, inp=[path, duration, offset], Tout=tf.string)

        return audio, transcript

    def process(self, dataset, batch_size: int):
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

        self.steps_per_epoch = len(dataset)

        return dataset

    @tf.function
    def parse(self, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            audio_signal, _ = tf.audio.decode_wav(
                audio, desired_channels=1, desired_samples=-1
            )
            audio_signal = tf.reshape(
                audio_signal, shape=[-1]
            )  # reshape for using tf.signal

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
