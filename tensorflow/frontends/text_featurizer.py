import os
import unicodedata
from typing import List, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tds


class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.beam_width = config.pop("beam_width", 0)
        self.blank_at_zero = config.pop("blank_at_zero", True)
        self.norm_score = config.pop("norm_score", True)
        self.lm_config = config.pop("lm_config", {})

        self.vocabulary = preprocess_paths(config.pop("vocabulary", None))
        self.target_vocab_size = config.pop("target_vocab_size", 1024)
        self.max_subword_length = config.pop("max_subword_length", 4)
        self.output_path_prefix = preprocess_paths(
            config.pop("output_path_prefix", None)
        )
        self.model_type = config.pop("model_type", None)
        self.corpus_files = preprocess_paths(config.pop("corpus_files", []))
        self.max_corpus_chars = config.pop("max_corpus_chars", None)
        self.reserved_tokens = config.pop("reserved_tokens", None)

        for k, v in config.items():
            setattr(self, k, v)


def preprocess_paths(paths: Union[List, str]):
    if isinstance(paths, list):
        return [
            path
            if path.startswith("gs://")
            else os.path.abspath(os.path.expanduser(path))
            for path in paths
        ]
    elif isinstance(paths, str):
        return (
            paths
            if paths.startswith("gs://")
            else os.path.abspath(os.path.expanduser(paths))
        )
    else:
        return None


class SubwordFeaturizer:
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, decoder_config: dict, subwords=None):
        """
        decoder_config = {
            "target_vocab_size": int,
            "max_subword_length": 4,
            "max_corpus_chars": None,
            "reserved_tokens": None,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        self.scorer = None
        self.decoder_config = DecoderConfig(decoder_config)
        self.tokens2indices = {}
        self.tokens = []

        self.subwords = self.__load_subwords() if subwords is None else subwords
        self.blank = 0  # subword treats blank as 0
        self.num_classes = self.subwords.vocab_size
        # create upoints
        self.__init_upoints()

    def __init_upoints(self):
        text = [""]
        for idx in np.arange(1, self.num_classes, dtype=np.int32):
            text.append(self.subwords.decode([idx]))
        self.upoints = tf.strings.unicode_decode(text, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_subword_length]

    def __load_subwords(self):
        filename_prefix = os.path.splitext(self.decoder_config.vocabulary)[0]
        return tds.deprecated.text.SubwordTextEncoder.load_from_file(filename_prefix)

    @classmethod
    def build_from_corpus(cls, decoder_config: dict, corpus_files: list = None):
        dconf = DecoderConfig(decoder_config.copy())
        corpus_files = (
            dconf.corpus_files
            if corpus_files is None or len(corpus_files) == 0
            else corpus_files
        )

        def corpus_generator():
            for file in corpus_files:
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    lines = lines[1:]
                for line in lines:
                    line = line.split("\t")
                    yield line[-1]

        subwords = tds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            corpus_generator(),
            dconf.target_vocab_size,
            dconf.max_subword_length,
            dconf.max_corpus_chars,
            dconf.reserved_tokens,
        )
        return cls(decoder_config, subwords)

    @classmethod
    def load_from_file(cls, decoder_config: dict, filename: str = None):
        dconf = DecoderConfig(decoder_config.copy())
        filename = dconf.vocabulary if filename is None else preprocess_paths(filename)
        filename_prefix = os.path.splitext(filename)[0]
        subwords = tds.deprecated.text.SubwordTextEncoder.load_from_file(
            filename_prefix
        )
        return cls(decoder_config, subwords)

    def preprocess_text(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        return text.strip("\n")  # remove trailing newline

    def add_scorer(self, scorer: any = None):
        """ Add scorer to this instance """
        self.scorer = scorer

    def normalize_indices(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Remove -1 in indices by replacing them with blanks
        Args:
            indices (tf.Tensor): shape any

        Returns:
            tf.Tensor: normalized indices with shape same as indices
        """
        with tf.name_scope("normalize_indices"):
            minus_one = -1 * tf.ones_like(indices, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
            return tf.where(indices == minus_one, blank_like, indices)

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """ Prepand blank index for transducer models """
        return tf.concat([[self.blank], text], axis=0)

    def save_to_file(self, filename: str = None):
        filename = (
            self.decoder_config.vocabulary
            if filename is None
            else preprocess_paths(filename)
        )
        filename_prefix = os.path.splitext(filename)[0]
        return self.subwords.save_to_file(filename_prefix)

    def extract(self, text: str) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self.preprocess_text(text)
        text = text.strip()  # remove trailing space
        indices = self.subwords.encode(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            total = tf.shape(indices)[0]
            batch = tf.constant(0, dtype=tf.int32)
            transcripts = tf.TensorArray(
                dtype=tf.string,
                size=total,
                dynamic_size=False,
                infer_shape=False,
                clear_after_read=False,
                element_shape=tf.TensorShape([]),
            )

            def cond(batch, total, _):
                return tf.less(batch, total)

            def body(batch, total, transcripts):
                upoints = self.indices2upoints(indices[batch])
                transcripts = transcripts.write(
                    batch, tf.strings.unicode_encode(upoints, "UTF-8")
                )
                return batch + 1, total, transcripts

            _, _, transcripts = tf.while_loop(
                cond, body, loop_vars=[batch, total, transcripts]
            )

            return transcripts.stack()

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))
