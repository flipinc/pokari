import io
import math
import os

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio

from augmentations import Augmentation

BUFFER_SIZE = 100
TFRECORD_SHARDS = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_and_convert_to_wav(path: str) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)


def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1:
            wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1:
            ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def tf_read_raw_audio(audio: tf.Tensor, sample_rate=16000):
    wave, rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    resampled = tfio.audio.resample(
        wave, rate_in=tf.cast(rate, dtype=tf.int64), rate_out=sample_rate
    )
    return tf.reshape(resampled, shape=[-1])  # reshape for using tf.signal


def get_num_batches(samples, batch_size, drop_remainders=True):
    if samples is None or batch_size is None:
        return None
    if drop_remainders:
        return math.floor(float(samples) / float(batch_size))
    return math.ceil(float(samples) / float(batch_size))


class SliceDataset:
    """ Keras Dataset for ASR using Slice """

    def __init__(
        self,
        stage: str,
        speech_featurizer,
        text_featurizer,
        data_paths: list,
        augmentations: Augmentation = Augmentation(None),
        cache: bool = False,
        shuffle: bool = False,
        use_tf: bool = False,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        **kwargs,
    ):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.data_paths = data_paths
        self.augmentations = augmentations  # apply augmentation
        self.cache = cache  # whether to cache WHOLE transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size  # shuffle buffer size
        self.stage = stage  # for defining tfrecords files
        self.use_tf = use_tf
        self.drop_remainder = (
            drop_remainder  # whether to drop remainder for multi gpu training
        )
        self.total_steps = None  # for better training visualization

    # def generator(self):
    #     for path, _, indices in self.entries:
    #         audio = load_and_convert_to_wav(path).numpy()
    #         yield bytes(path, "utf-8"), audio, bytes(indices, "utf-8")

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)

    def read_entries(self):
        self.entries = []
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                self.entries += temp_lines
        # The files is "\t" seperated
        self.entries = [line.split("\t", 1) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join(
                [str(x) for x in self.text_featurizer.extract(line[-1]).numpy()]
            )
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes):
            return load_and_convert_to_wav(path.decode("utf-8")).numpy()

        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        return record[0], audio, record[1]

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                {
                    "path": tf.TensorShape([]),
                    "input": tf.TensorShape(self.speech_featurizer.shape),
                    "input_length": tf.TensorShape([]),
                    "prediction": tf.TensorShape([None]),
                    "prediction_length": tf.TensorShape([]),
                },
                {"label": tf.TensorShape([None]), "label_length": tf.TensorShape([])},
            ),
            padding_values=(
                {
                    "path": "",
                    "input": 0.0,
                    "input_length": 0,
                    "prediction": self.text_featurizer.blank,
                    "prediction_length": 0,
                },
                {"label": self.text_featurizer.blank, "label_length": 0},
            ),
            drop_remainder=self.drop_remainder,
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(
            self.total_steps, batch_size, drop_remainders=self.drop_remainder
        )
        return dataset

    @tf.function
    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        data = self.tf_preprocess(path, audio, indices)

        (
            path,
            features,
            input_length,
            label,
            label_length,
            prediction,
            prediction_length,
        ) = data

        return (
            {
                "path": path,
                "input": features,
                "input_length": input_length,
                "prediction": prediction,
                "prediction_length": prediction_length,
            },
            {"label": label, "label_length": label_length},
        )

    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)

            signal = self.augmentations.before.augment(signal)

            features = self.speech_featurizer.tf_extract(signal)

            features = self.augmentations.after.augment(features)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)
            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)
            features = tf.convert_to_tensor(features, tf.float32)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            return (
                path,
                features,
                input_length,
                label,
                label_length,
                prediction,
                prediction_length,
            )
