import abc
import os
import re
import unicodedata
from typing import List, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tds

import frontends.wordpiece as wordpiece
from frontends.character import en, jp


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
    def __init__(self):
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None

    def preprocess_text(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        return text.strip("\n")  # remove trailing newline

    def normalize_indices(self, indices: tf.Tensor) -> tf.Tensor:
        """Remove -1 in indices by replacing them with blanks
        Args:
            indices (tf.Tensor): shape any

        Returns:
            (tf.Tensor): normalized indices with shape same as indices
        """
        minus_one = -1 * tf.ones_like(indices, dtype=tf.int32)
        blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
        return tf.where(indices == minus_one, blank_like, indices)

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """Prepand blank index for transducer models"""
        return tf.concat([[self.blank], text], axis=0)

    @abc.abstractclassmethod
    def init_upoints(self):
        """Create a dictionary of upoints that map to target indicies"""
        raise NotImplementedError()

    @abc.abstractclassmethod
    def string2Indices(self, text):
        """Converts a string to a list of indices

        Args:
            text (string): sequence of characters

        Returns:
            (tf.Tensor): sequence of ints
        """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def indices2String(self, indices):
        """Directly converts a list of indices to string

        Note: this function is intended for only debugging purpose

        Args:
            indices (tf.Tensor): [B, None]

        Returns:
            (tf.Tensor): transcripts of dtype tf.string with dim [B]
        """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def indices2upoints(self, indices):
        """Transforms predicted indices to unicode code points (for using tflite)

        Args:
            indices (tf.Tensor): indices in shape [None] or [B, None]

        Returns:
            (tf.Tensor): unicode code points transcript with dtype tf.int32 and shape
            [None] or [B, None]

        """
        raise NotImplementedError()


class CharFeaturizer(TextFeaturizer):
    def __init__(
        self,
        vocab_path: str = None,
        blank_at_zero: bool = True,
        lang: str = "en",
        normalize: bool = False,
    ):
        super().__init__()

        self.vocab_path = vocab_path
        self.blank_at_zero = blank_at_zero
        self.lang = lang
        self.normalize = normalize

        self.init_vocabulary()
        self.init_upoints()

    def init_vocabulary(self):
        lines = []
        if self.vocab_path is not None:
            with open(self.vocab_path, "r", encoding="utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            if self.lang == "en":
                lines = en
            elif self.lang == "jp":
                lines = jp
            else:
                raise ValueError("This language is not supported.")

        self.blank = 0 if self.blank_at_zero else None
        self.tokens2indices = {}
        self.tokens = []

        index = 1 if self.blank == 0 else 0
        for line in lines:
            if self.normalize:
                line = self.preprocess_text(line)

            self.tokens2indices[line[0]] = index
            self.tokens.append(line[0])
            index += 1

        if self.blank is None:
            self.blank = len(self.tokens)  # blank not at zero

        self.vocab_array = self.tokens.copy()
        self.tokens.insert(self.blank, "")  # add blank token to tokens
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)

    def init_upoints(self):
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(
            shape=[None, 1]
        )

    def string2Indices(self, text: str) -> tf.Tensor:
        if self.normalize:
            text = self.preprocess_text(text)

        text = list(text.strip())  # remove trailing space
        indices = [self.tokens2indices[token] for token in text]
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def indices2String(self, indices: tf.Tensor) -> tf.Tensor:
        indices = self.normalize_indices(indices)
        tokens = tf.gather_nd(self.tokens, tf.expand_dims(indices, axis=-1))
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            tokens = tf.strings.reduce_join(tokens, axis=-1)
        return tokens

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(self, indices: tf.Tensor) -> tf.Tensor:
        indices = self.normalize_indices(indices)
        upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
        return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))


class WordpieceFeaturizer(TextFeaturizer):
    PAD_TOKEN = "<pad>"  # also used as blank token

    def __init__(
        self,
        subwords_path: str,
        subwords_corpus: str = None,
        target_vocab_size: int = 1024,
        max_token_length: int = 4,
        normalize: bool = False,
    ):
        """Text featurizer based on Wordpiece

        TODO: Since Japanese is the top priority as of 2021/3 and japanese asr performs
        better in character (see refs), wordpiece in production is not supported,
        althought it is just fine in training. This needs to be exportable to TFLite!!
        ref: https://github.com/espnet/espnet/pull/326
        ref: http://www.interspeech2020.org/uploadfile/pdf/Mon-2-2-2.pdf

        """
        super().__init__()

        self.target_vocab_size = target_vocab_size
        self.max_token_length = max_token_length
        self.max_corpus_chars = None
        self.reserved_tokens = [self.PAD_TOKEN]

        self.normalize = normalize

        if subwords_path and os.path.exists(subwords_path):
            self.subwords = self.load_from_file(subwords_path)
        else:
            self.subwords = self.build_from_corpus(subwords_corpus, subwords_path)

        self.blank = 0  # subword treats blank as 0
        self.num_classes = self.subwords.vocab_size

        self.init_upoints()

    def init_upoints(self):
        text = [""]  # first one is reserved for blank token
        # start iteration after blank
        for idx in np.arange(1, self.num_classes, dtype=np.int32):
            token = self.subwords.detokenize(tf.constant([[idx]], tf.int32))
            token = token.values.numpy()[0].decode("utf-8")

            # `##_` in wordpiece means, a beginging space should be trimmed, otherwise
            # prepend a space
            if "#" in token:
                token = re.sub("#", "", token)
            else:
                token = " " + token

            text.append(token)
        self.upoints = tf.strings.unicode_decode(text, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_token_length]

    def build_from_corpus(self, corpus_files: list = None, output_file: str = None):
        filename = preprocess_paths(output_file)

        def corpus_generator():
            for file in corpus_files:
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    lines = lines[1:]
                for line in lines:
                    line = line.split("\t")
                    yield line[-1]

        wordpiece.build_from_corpus(
            corpus_generator(),
            output_file_path=filename,
            target_vocab_size=self.target_vocab_size,
            max_token_length=self.max_token_length,
            max_corpus_chars=self.max_corpus_chars,
            reserved_tokens=self.reserved_tokens,
        )

        subwords = wordpiece.WordpieceTokenizer(filename, token_out_type=tf.int32)
        return subwords

    def load_from_file(self, filename: str = None):
        filename = preprocess_paths(filename)
        subwords = wordpiece.WordpieceTokenizer(filename, token_out_type=tf.int32)
        return subwords

    def string2Indices(self, text: str) -> tf.Tensor:
        if self.normalize:
            text = self.preprocess_text(text)

        # remove trailing space
        text = text.strip()
        splitted_text = text.split()
        indices = self.subwords.tokenize(splitted_text)
        # `tokenize` returns ragged tensor
        indices = indices.merge_dims(0, -1)
        return tf.cast(indices, dtype=tf.int32)

    def indices2String(self, indices: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            indices = self.normalize_indices(indices)
            # returns ragged tensor with one element corresponds to one word
            # eg) ["i'm", "<pad>", "thirsty", "<pad>", ...]
            text = self.subwords.detokenize(indices)
            # ["i'm", "", "thirsty", ""]
            text = tf.strings.regex_replace(text, self.PAD_TOKEN, "")
            # ["i'm  thirsty "] <- double spaces & trailing space
            text = tf.strings.reduce_join(text, separator=" ", axis=-1)
            # ["i'm thirsty "] <- trailing space
            text = tf.strings.regex_replace(text, " +", " ")
            # ["i'm thirsty"]
            text = tf.strings.strip(text)
            return text

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def indices2upoints(self, indices: tf.Tensor) -> tf.Tensor:
        indices = self.normalize_indices(indices)
        upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
        return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))


class DeprecatedWordpieceFeaturizer(TextFeaturizer):
    def __init__(
        self,
        subwords_path: str,
        subwords_corpus: str = None,
        target_vocab_size: int = 1024,
        max_token_length: int = 4,
        normalize: bool = False,
    ):
        super().__init__()

        self.target_vocab_size = target_vocab_size
        self.max_token_length = max_token_length
        self.max_corpus_chars = None
        # unlike undeprecated version, tds.deprecated.text.SubwordTextEncoder
        # automatically register blank token to zero index
        self.reserved_tokens = None

        self.normalize = normalize

        if subwords_path and os.path.exists(subwords_path):
            self.subwords = self.load_from_file(subwords_path)
        else:
            self.subwords = self.build_from_corpus(subwords_corpus)
            self.save_to_file(subwords_path)

        self.blank = 0  # subword treats blank as 0
        self.num_classes = self.subwords.vocab_size

        self.init_upoints()

    def init_upoints(self):
        text = [""]  # first one is reserved for blank token
        for idx in np.arange(1, self.num_classes, dtype=np.int32):
            token = self.subwords.decode([idx])
            text.append(token)
        self.upoints = tf.strings.unicode_decode(text, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_token_length]

    def build_from_corpus(self, corpus_files: list = None):
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
            self.target_vocab_size,
            self.max_token_length,
            self.max_corpus_chars,
            self.reserved_tokens,
        )
        return subwords

    def load_from_file(self, filename: str = None):
        filename = preprocess_paths(filename)
        filename_prefix = os.path.splitext(filename)[0]
        subwords = tds.deprecated.text.SubwordTextEncoder.load_from_file(
            filename_prefix
        )
        return subwords

    def save_to_file(self, filename: str = None):
        filename = preprocess_paths(filename)
        filename_prefix = os.path.splitext(filename)[0]
        return self.subwords.save_to_file(filename_prefix)

    def string2Indices(self, text: str) -> tf.Tensor:
        if self.normalize:
            text = self.preprocess_text(text)

        text = text.strip()  # remove trailing space
        indices = self.subwords.encode(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def indices2String(self, indices: tf.Tensor) -> tf.Tensor:
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
        indices = self.normalize_indices(indices)
        upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
        return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))
