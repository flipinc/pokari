import abc
import os
import unicodedata
from typing import List, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tds


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


class TextFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self, decoder_config: dict):
        self.scorer = None
        self.decoder_config = decoder_config
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None
        self.max_length = 0

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

    @abc.abstractclassmethod
    def extract(self, text):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def iextract(self, indices):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def indices2upoints(self, indices):
        raise NotImplementedError()


class CharFeaturizer(TextFeaturizer):
    def __init__(
        self,
        vocabulary: str = None,
        blank_at_zero: bool = True,
    ):
        decoder_config = {
            "vocabulary": vocabulary,
            "blank_at_zero": blank_at_zero,
        }
        super().__init__(decoder_config)
        self.__init_vocabulary()

    def __init_vocabulary(self):
        lines = []
        if self.decoder_config["vocabulary"] is not None:
            with open(self.decoder_config["vocabulary"], "r", encoding="utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = [
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
            ]
        self.blank = 0 if self.decoder_config["blank_at_zero"] else None
        self.tokens2indices = {}
        self.tokens = []
        index = 1 if self.blank == 0 else 0
        for line in lines:
            line = self.preprocess_text(line)
            if line.startswith("#") or not line:
                continue
            self.tokens2indices[line[0]] = index
            self.tokens.append(line[0])
            index += 1
        if self.blank is None:
            self.blank = len(self.tokens)  # blank not at zero
        self.vocab_array = self.tokens.copy()
        self.tokens.insert(self.blank, "")  # add blank token to tokens
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(
            shape=[None, 1]
        )

    def extract(self, text: str) -> tf.Tensor:
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self.preprocess_text(text)
        text = list(text.strip())  # remove trailing space
        indices = [self.tokens2indices[token] for token in text]
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        tokens = tf.gather_nd(self.tokens, tf.expand_dims(indices, axis=-1))
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            tokens = tf.strings.reduce_join(tokens, axis=-1)
        return tokens

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


class SubwordFeaturizer(TextFeaturizer):
    def __init__(
        self,
        subwords_path: str,
        subwords_corpus: str = None,
        vocabulary: str = None,
        target_vocab_size: int = 1024,
        max_subword_length: int = 4,
    ):
        decoder_config = {
            "vocabulary": vocabulary,
            "target_vocab_size": target_vocab_size,
            "max_subword_length": max_subword_length,
            "max_corpus_chars": None,
            "reserved_tokens": None,
        }
        super().__init__(decoder_config)

        if subwords_path and os.path.exists(subwords_path):
            self.subwords = self.load_from_file(decoder_config, subwords_path)
        else:
            self.subwords = self.build_from_corpus(decoder_config, subwords_corpus)
            self.save_to_file(subwords_path)

        # self.subwords = self.__load_subwords() if subwords is None else subwords
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

    # def __load_subwords(self):
    #     filename_prefix = os.path.splitext(self.decoder_config["vocabulary"])[0]
    #     return tds.deprecated.text.SubwordTextEncoder.load_from_file(filename_prefix)

    def build_from_corpus(self, dconf: dict, corpus_files: list = None):
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
            dconf["target_vocab_size"],
            dconf["max_subword_length"],
            dconf["max_corpus_chars"],
            dconf["reserved_tokens"],
        )
        return subwords

    def load_from_file(self, dconf: dict, filename: str = None):
        filename = (
            dconf["vocabulary"] if filename is None else preprocess_paths(filename)
        )
        filename_prefix = os.path.splitext(filename)[0]
        subwords = tds.deprecated.text.SubwordTextEncoder.load_from_file(
            filename_prefix
        )
        return subwords

    def save_to_file(self, filename: str = None):
        filename = (
            self.decoder_config["vocabulary"]
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
