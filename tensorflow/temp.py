import multiprocessing
import os
import sys

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

BUFFER_SIZE = 100
TFRECORD_SHARDS = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE


def print_one_line(*args):
    tf.print("\033[K", end="")
    tf.print("\r", *args, sep="", end=" ", output_stream=sys.stdout)


def bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def load_and_convert_to_wav(path: str) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)


def tf_read_raw_audio(audio: tf.Tensor, sample_rate=16000):
    wave, rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    resampled = tfio.audio.resample(
        wave, rate_in=tf.cast(rate, dtype=tf.int64), rate_out=sample_rate
    )
    return tf.reshape(resampled, shape=[-1])  # reshape for using tf.signal


class ASRTFRecordDatasetKeras(object):
    """ Keras Dataset for ASR using TFRecords """

    def __init__(
        self,
        data_paths: list,
        tfrecords_dir: str,
        speech_featurizer: SpeechFeaturizer,
        text_featurizer: TextFeaturizer,
        stage: str,
        augmentations: Augmentation = Augmentation(None),
        tfrecords_shards: int = TFRECORD_SHARDS,
        cache: bool = False,
        shuffle: bool = False,
        use_tf: bool = False,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        **kwargs,
    ):

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

        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

        if not self.stage:
            raise ValueError("stage must be defined, either 'train', 'eval' or 'test'")
        self.tfrecords_dir = tfrecords_dir
        if tfrecords_shards <= 0:
            raise ValueError("tfrecords_shards must be positive")
        self.tfrecords_shards = tfrecords_shards
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

    def read_entries(self):
        self.entries = []
        for file_path in self.data_paths:
            print(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                self.entries += temp_lines[1:]
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join(
                [str(x) for x in self.text_featurizer.extract(line[-1]).numpy()]
            )
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)

    @staticmethod
    def write_tfrecord_file(splitted_entries):
        shard_path, entries = splitted_entries
        with tf.io.TFRecordWriter(shard_path, options="ZLIB") as out:
            for path, _, indices in entries:
                audio = load_and_convert_to_wav(path).numpy()
                feature = {
                    "path": bytestring_feature([bytes(path, "utf-8")]),
                    "audio": bytestring_feature([audio]),
                    "indices": bytestring_feature([bytes(indices, "utf-8")]),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                out.write(example.SerializeToString())
                print_one_line("Processed:", path)
        print(f"\nCreated {shard_path}")

    def create_tfrecords(self):
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

        if tf.io.gfile.glob(
            os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        ):
            print(f"TFRecords're already existed: {self.stage}")
            return True

        print(f"Creating {self.stage}.tfrecord ...")

        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return False

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f"{self.stage}_{shard_id}.tfrecord")

        shards = [get_shard_path(idx) for idx in range(1, self.tfrecords_shards + 1)]

        splitted_entries = np.array_split(self.entries, self.tfrecords_shards)
        with multiprocessing.Pool(self.tfrecords_shards) as pool:
            pool.map(self.write_tfrecord_file, zip(shards, splitted_entries))

        return True

    def create(self, batch_size: int):
        have_data = self.create_tfrecords()
        if not have_data:
            return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(
            files_ds, compression_type="ZLIB", num_parallel_reads=AUTOTUNE
        )

        return self.process(dataset, batch_size)  # generator

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
    def parse(self, record: tf.Tensor):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string),
        }
        path, audio, indices = tf.io.parse_single_example(record, feature_description)

        data = self.preprocess(path, audio, indices)

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

    def preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)

            signal = self.augmentations.before.augment(signal)

            features = self.speech_featurizer.tf_extract(signal)  # mel spectrogram

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
