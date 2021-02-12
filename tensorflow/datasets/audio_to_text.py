import multiprocessing
import os
from typing import Callable, List, Optional, Union

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tqdm import tqdm

from datasets.audio_augment import AudioAugmentor
from datasets.manifest import collect_manifest
from datasets.text_parser import make_parser


class DatasetCreator:
    """

    Args:
        drop_remainder: whether to drop remainder for multi gpu training
        trim: whether to trim silence

    """

    def __init__(
        self,
        batch_size: int,
        stage: str,
        tfrecords_dir: str,
        manifest_filepath: str,
        labels: List[str],
        sample_rate: int,
        cache: bool = True,
        shuffle: bool = True,
        buffer_size: int = 100,
        tfrecords_shards: int = 16,
        int_values: bool = False,
        drop_remainder: bool = False,
        augmentor=None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        ignore_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = "en",
    ):
        self.labels = labels
        self.sample_rate = sample_rate
        self.batch_size = batch_size

        self.stage = stage
        self.tfrecords_dir = tfrecords_dir
        self.tfrecords_shards = tfrecords_shards
        self.cache = cache
        self.shuffle = shuffle
        self.buffer_size = buffer_size

        self.int_values = int_values
        self.drop_remainder = drop_remainder
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()

        self.parser = make_parser(
            labels=labels,
            name=parser,
            unk_id=unk_index,
            ignore_id=ignore_index,
            do_normalize=normalize,
        )

        self.collections = collect_manifest(
            manifests_files=manifest_filepath.split(","),
            parser=self.parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

    def create(self):
        self.create_tfrecords()

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files = tf.data.Dataset.list_files(pattern)

        # experimental_deterministic should be set to False
        # ref: https://github.com/tensorflow/datasets/issues/951
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False

        dataset = tf.data.TFRecordDataset(
            files,
            compression_type="ZLIB",
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.map(
            self.collate_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=(
                tf.TensorShape([None]),  # audio
                tf.TensorShape([]),  # audio_len
                tf.TensorShape([None]),  # transcript
                tf.TensorShape([]),  # transcript_len
            ),
            padding_values=(
                0.0,
                0,
                self.pad_id,
                0,
            ),
            drop_remainder=self.drop_remainder,
        )

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def collate_fn(self, tf_record: tf.Tensor):
        result = tf.io.parse_single_example(
            tf_record,
            features={
                "audio": tf.io.FixedLenFeature([], tf.string),
                "text_tokens": tf.io.FixedLenFeature([], tf.string),
            },
        )

        audio = result["audio"]
        audio, sample_rate = tf.audio.decode_wav(
            audio, desired_channels=1, desired_samples=-1
        )
        audio = tfio.audio.resample(
            audio,
            rate_in=tf.cast(sample_rate, dtype=tf.int64),
            rate_out=self.sample_rate,
        )
        audio_len = tf.cast(tf.shape(audio)[0], tf.int32)

        transcript = result["text_tokens"]
        transcript = tf.strings.to_number(
            tf.strings.split(transcript), out_type=tf.int32
        )
        transcript_len = tf.cast(len(transcript), tf.int32)

        with tf.device("/CPU:0"):
            audio = self.augmentor.perturb(audio)

            token_length = len(transcript)
            if self.bos_id is not None:
                transcript = [self.bos_id] + transcript
                token_length += 1
            if self.eos_id is not None:
                transcript = transcript + [self.eos_id]
                token_length += 1

            return audio, audio_len, transcript, transcript_len

    def create_tfrecords(self):
        if not tf.io.gfile.exists(self.tfrecords_dir):
            tf.io.gfile.makedirs(self.tfrecords_dir)

        if tf.io.gfile.glob(
            os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        ):
            return

        shard_paths = [
            os.path.join(self.tfrecords_dir, f"{self.stage}_{idx}.tfrecord")
            for idx in range(1, self.tfrecords_shards + 1)
        ]

        splitted_collections = np.array_split(
            np.array(self.collections), self.tfrecords_shards
        )
        with multiprocessing.Pool(self.tfrecords_shards) as pool:
            pool.map(self.create_one_tfrecord, zip(shard_paths, splitted_collections))

    def create_one_tfrecord(self, inputs):
        shard_path, splitted_collection = inputs
        with tf.io.TFRecordWriter(shard_path, options="ZLIB") as out:
            for (
                id,
                audio_file,
                duration,
                text_tokens,
                offset,
                text_raw,
                speaker,
                orig_sr,
            ) in tqdm(splitted_collection):
                wave, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                audio = tf.audio.encode_wav(
                    tf.expand_dims(wave, axis=-1), sample_rate=sample_rate
                )

                encoded_text_tokens = " ".join([str(token) for token in text_tokens])

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "audio": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[audio.numpy()])
                            ),
                            "label": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[bytes(encoded_text_tokens, "utf-8")]
                                )
                            ),
                        }
                    )
                )

                out.write(example.SerializeToString())

            out.close()
