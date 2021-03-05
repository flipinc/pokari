import math
from typing import Callable, Union

import numpy as np
import tensorflow as tf

from modules.subsample import StackSubsample, VggSubsample


class EmformerEncoder(tf.keras.Model):
    """

    Some training tips for emformer
    - (Only confirmed in PyTorch) to keep positional information, vgg
    subsampling is preferred over stack subsampling

    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        dim_model: int,
        dim_ffn: int,
        dropout_attn: int,
        dropout_ffn: int,
        subsampling: str,
        subsampling_factor: int,
        subsampling_dim: int,
        left_length: int,
        chunk_length: int,
        right_length: int,
        name: str = "emformer_encoder",
    ):
        super().__init__(name=name)

        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.left_length = left_length
        self.chunk_length = chunk_length
        self.right_length = right_length
        self.dim_model = dim_model

        feat_out = int(dim_model / subsampling_factor)
        if dim_model % subsampling_factor > 0:
            raise ValueError("dim_model must be divisible by subsampling_factor.")

        self.linear = tf.keras.layers.Dense(feat_out)

        if self.subsampling == "stack":
            self.subsample = StackSubsample(
                subsampling_factor=self.subsampling_factor,
                name=f"{self.name}_stack_subsample",
            )
        elif self.subsampling == "vgg":
            self.subsample = VggSubsample(
                subsampling_factor=self.subsampling_factor,
                feat_out=dim_model,
                conv_channels=subsampling_dim,
                name=f"{self.name}_vgg_subsample",
            )

        self.blocks = []
        for i in range(self.num_layers):
            block = EmformerBlock(
                num_heads=self.num_heads,
                dim_model=self.dim_model,
                dim_ffn=dim_ffn,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn,
                left_length=self.left_length,
                chunk_length=self.chunk_length,
                right_length=self.right_length,
                name=f"{self.name}_{i}_block",
            )
            self.blocks.append(block)

    def create_mask(self, audio_lens: np.array, t_max: int):
        """
        Numpy implementatino of emformer attention mask. Only used during training.
        In graph mode this is as fast as numpy impl. But using this in eager mode
        is not recommended.

        There are four types of masks.
        - mask_body: A attention mask for every timestep.
        - mask_left: A attention mask for every timestep. Hard copied
            right contexts included.
        - mark_right: A attention mask for hard copied right contexts.
        - mark_diagnal: A attention mask for hard copied right contexts
            with copied right contexts.

        Note: If you want to see Tensorflow impl, see commit hash
        a45040e9e6726d48e7d5e50be88a2bd9da65fb0f

        Args:
            audio_lens (tf.Tensor): [B]
            t_max (int): this cannot be inferred by audio_lens because longest input_len
                may be padded for efficiency
        """
        audio_lens = audio_lens.astype("int32")

        bs = audio_lens.shape[0]
        num_chunks = math.ceil(t_max / self.chunk_length)

        # TODO: this is allocating more than what is actually required
        upperbound_right_length = (num_chunks - 1) * self.right_length

        mask_body = np.zeros([bs, t_max, t_max])
        mask_left = np.zeros([bs, t_max, upperbound_right_length])
        mask_diagnal = np.zeros([bs, upperbound_right_length, upperbound_right_length])
        mask_right = np.zeros([bs, upperbound_right_length, t_max])

        right_indexes = np.zeros(0).astype("int32")
        for i in range(num_chunks):
            # 1. mark diagnal and left chunks
            left_offset = (
                self.chunk_length * i - self.left_length
                if (self.chunk_length * i - self.left_length) >= 0
                else 0
            )
            start_offset = self.chunk_length * i
            end_offset = (
                self.chunk_length * (i + 1)
                if self.chunk_length * (i + 1) <= t_max
                else t_max
            )
            mask_body[:, start_offset:end_offset, left_offset:end_offset] = 1

            # last segment does not have right context
            if i == (num_chunks - 1):
                break

            # 2. hard copy right contexts
            remaining_space = t_max - (end_offset + self.right_length)
            if remaining_space >= 0:
                this_right_length = self.right_length
            else:
                this_right_length = t_max - end_offset

            mask_left[
                :,
                start_offset : start_offset + self.chunk_length,
                len(right_indexes) : len(right_indexes) + this_right_length,
            ] = 1

            mask_diagnal[
                :,
                len(right_indexes) : len(right_indexes) + this_right_length,
                len(right_indexes) : len(right_indexes) + this_right_length,
            ] = 1

            mask_right[
                :,
                len(right_indexes) : len(right_indexes) + this_right_length,
                left_offset:end_offset,
            ] = 1

            right_indexes = np.concatenate(
                [
                    right_indexes,
                    tf.range(end_offset, end_offset + this_right_length),
                ]
            ).astype("int32")

        # 3. remove unused right contexts
        right_size = len(right_indexes)
        mask_left = mask_left[:, :, :right_size]
        mask_diagnal = mask_diagnal[:, :right_size, :right_size]
        mask_right = mask_right[:, :right_size, :]

        # 4. mask paddings
        # TODO: there should be a way to parallelize this
        for i in range(bs):
            max_len = audio_lens[i]

            # 4.1 pad mask_body and mask_right
            mask_body[i, max_len:, :] = 0
            mask_body[i, :, max_len:] = 0
            mask_right[i, :, max_len:] = 0

            to_be_padded = right_indexes[np.nonzero(right_indexes >= max_len)]
            if to_be_padded.shape[0] > 0:
                # retrieve first index of right index that needs padding (int)
                pad_begin_index = np.where(right_indexes == int(to_be_padded[0]))[0][0]

                # 4.2 pad mask_left and mask_diagnal and mask_right
                mask_right[i, pad_begin_index:, :] = 0
                mask_left[i, :, pad_begin_index:] = 0
                mask_diagnal[i, pad_begin_index:, :] = 0
                mask_diagnal[i, :, pad_begin_index:] = 0

        # 5. concatenate all masks
        mask_top = np.concatenate([mask_diagnal, mask_right], axis=-1)
        mask_bottom = np.concatenate([mask_left, mask_body], axis=-1)
        mask = np.concatenate([mask_top, mask_bottom], axis=-2)
        mask = np.expand_dims(mask, axis=1)

        return mask.astype("float32"), right_indexes

    def stream(
        self,
        audio_features: tf.Tensor,
        cache: tf.Tensor,
    ):
        """
        Note: cache_k and cache_v must be given. These can be retrieved by calling
        get_initial_state().

        Note: streaming can be done in batch, however, all batch must have same length.

        Args:
            cache: two for cache_k and cache_v. [2, N, B, L, H, D]

        """
        # 1. projection
        x = self.linear(audio_features)

        # 2. subsampling
        x = self.subsample(x)

        # 3. loop over blocks while saving cache at the same time
        new_cache_k = new_cache_v = []
        for i, block in enumerate(self.blocks):
            x, (temp_cache_k, temp_cache_v) = block.stream(x, cache[0][i], cache[1][i])
            new_cache_k.append(temp_cache_k)
            new_cache_v.append(temp_cache_v)

        new_cache_k = tf.stack(new_cache_k, axis=0)
        new_cache_v = tf.stack(new_cache_v, axis=0)
        new_cache = tf.stack([new_cache_k, new_cache_v], axis=0)

        # 4. Trim right context
        x = x[:, : self.chunk_length, :]

        return x, new_cache

    def get_initial_state(self, batch_size: int):
        """Get initial state for streaming emformer [2, N, B, L, H, D/H]"""
        return tf.zeros(
            [
                2,  # cache_k and cache_v
                self.num_layers,
                batch_size,
                self.left_length,
                self.num_heads,
                self.dim_model // self.num_heads,
            ]
        )

    def call(
        self, audio_features: tf.Tensor, audio_lens: tf.int32, training: bool = None
    ):
        """

        D_1: number of mels. This number has to match D_2 after frame stacking
        D_2: encoder dim.

        Args:
            audio_features (tf.Tensor): [B, Tmax, D_1]
            audio_lens (tf.int32): [B]

        Returns:
            tf.Tensor: [B, Tmax, D_2]
        """
        # 1. projection
        x = self.linear(audio_features)

        # 2. subsampling
        x, audio_lens = self.subsample(x, audio_lens)

        # 3. create attention mask
        t_new = tf.cast(tf.shape(x)[1], tf.int32)
        # using numpy mask instead of tensorflow gives 40%+ training time reduction
        # and this does not affect training at all
        mask, right_indexes = tf.numpy_function(
            self.create_mask, [audio_lens, t_new], [tf.float32, tf.int32]
        )
        # need to explicitly set shape for numpy mask, otherwise build() fails
        mask.set_shape([None, 1, None, None])  # [B, 1, Total_R + Tmax, Total_R + Tmax]

        # 4. Hard copy right context and prepare input for the first iteration
        # [B, Total_R+Tmax, D]
        x_right = tf.gather(x, right_indexes, axis=1)
        x = tf.concat([x_right, x], axis=1)

        # 5. loop over blocks.
        for block in self.blocks:
            x = block(x, mask=mask, training=training)

        # 6. Trim copied right context
        x = x[:, len(right_indexes) :, :]

        return x, audio_lens


class EmformerBlock(tf.keras.layers.Layer):
    """Emformer Block
    Emformer Blocks are just custom impl of multi-head attention. See
    https://arxiv.org/pdf/1910.12977.pdf for more detail.
    Code is referenced from tfa.layers.MultiHeadAttention
    """

    def __init__(
        self,
        num_heads: int,
        dim_model: int,
        dim_ffn: int,
        dropout_attn: int,
        dropout_ffn: int,
        left_length: int,
        chunk_length: int,
        right_length: int,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        bias_regularizer: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        name: str = "emformer_block",
    ):
        super().__init__(name=name)

        self.left_length = left_length
        self.chunk_length = chunk_length
        self.right_length = right_length

        self.head_size = dim_model // num_heads
        self.num_heads = num_heads

        kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        bias_initializer = tf.keras.initializers.get(bias_initializer)
        bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.query_kernel = self.add_weight(
            name=f"{name}_query_kernel",
            shape=[num_heads, dim_model, self.head_size],
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            constraint=kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name=f"{name}_key_kernel",
            shape=[num_heads, dim_model, self.head_size],
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            constraint=kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name=f"{name}_value_kernel",
            shape=[num_heads, dim_model, self.head_size],
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            constraint=kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name=f"{name}_projection_kernel",
            shape=[num_heads, self.head_size, dim_model],
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            constraint=kernel_constraint,
        )
        self.projection_bias = self.add_weight(
            name=f"{name}_projection_bias",
            shape=[dim_model],
            initializer=bias_initializer,
            regularizer=bias_regularizer,
            constraint=bias_constraint,
        )

        self.ln_in = tf.keras.layers.LayerNormalization()
        self.ln_out_1 = tf.keras.layers.LayerNormalization()
        self.ln_out_2 = tf.keras.layers.LayerNormalization()

        self.attn_dropout = tf.keras.layers.Dropout(dropout_attn)

        self.linear_dropout_1 = tf.keras.layers.Dropout(dropout_ffn)
        self.linear_dropout_2 = tf.keras.layers.Dropout(dropout_ffn)

        self.linear_out_1 = tf.keras.layers.Dense(dim_ffn, activation="relu")
        self.linear_out_2 = tf.keras.layers.Dense(dim_model)

    def attend(
        self,
        x: tf.Tensor,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = None,
    ):
        """

        N: Total_R + Tmax at training time. segment_size at inference time.
        M: Total_R + Tmax at training time. segment_size + left_size at inference time.

        Args:
            q: [B, N, H, D]
            k: [B, M, H, D]
            v: [B, M, H, D]
            mask: Only used at training time. [B, 1, M, M]

        """
        # 1. get attention scores -> [B, H, N, M]
        # doing the division to either query or key instead of their product saves
        # some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        q /= tf.sqrt(depth)
        attn_scores = tf.einsum("...NHO,...MHO->...HNM", q, k)

        # 2. apply mask and softmax
        if mask is not None:
            # using float(-inf) or -math.inf gives nan after softmax
            attn_scores += -10e9 * (1.0 - mask)
            attn_probs = tf.nn.softmax(attn_scores, axis=-1)
            # TODO: does doing this improve accuracy?
            attn_probs = attn_probs * mask
        else:
            attn_probs = tf.nn.softmax(attn_scores, axis=-1)

        attn_probs = self.attn_dropout(attn_probs, training=training)

        # 3. attention * value
        output = tf.einsum("...HNM,...MHI->...NHI", attn_probs, v)

        # 4. calculate dot product attention
        output = tf.einsum("...NHI,HIO->...NO", output, self.projection_kernel)
        output += self.projection_bias

        attn_out = output + x

        # 5. FFN module in transformer
        output = self.ln_out_1(attn_out, training=training)

        output = self.linear_out_1(output)
        output = self.linear_dropout_1(output, training=training)
        output = self.linear_out_2(output)
        output = self.linear_dropout_2(output, training=training)

        output += attn_out

        output = self.ln_out_2(output, training=training)

        return output

    def stream(
        self,
        inputs: tf.Tensor,
        cache_k: tf.Tensor,
        cache_v: tf.Tensor,
    ):
        """
        Note: At inference time, the notion of Left, Chunk, Right contexts still exists
        because without them, the compatibility with training will be lost.
        For example, at training time, the output for t_0, t_1, t_2 is determined by
        the input of t_0, t_1, t_2 PLUS right contexts. This must be the case for
        inference too.

        Note: Only a part of chunk context is cached as next left context. Right context
        cannot be cached because, at training time, right context at layer n is
        calculated based on current segment (not previous segment).

        N: number of layers
        S: segment size (chunk size + right size)
        M: segment size with left context (segment size + left size)

        Args:
            x (tf.Tensor): [B, S, D]
            cache_k (tf.Tensor): [N, B, L, H, D/H]
            cache_v (tf.Tensor): [N, B, L, H, D/H]
        """
        # 1. apply layer norm
        x = self.ln_in(inputs)

        # 2. calculate q -> [B, S, H, D]
        q = tf.einsum("...SI,HIO->...SHO", x, self.query_kernel)

        # 3. calculate k -> [B, M, H, D]
        k_cr = tf.einsum("...SI,HIO->...SHO", x, self.key_kernel)
        k = tf.concat([cache_k, k_cr], axis=1)
        new_cache_k = k[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]

        # 4. calculate v -> [B, M, H, D]
        v_cr = tf.einsum("...SI,HIO->...SHO", x, self.value_kernel)
        v = tf.concat([cache_v, v_cr], axis=1)
        new_cache_v = v[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]

        output = self.attend(inputs, q, k, v, training=False)

        return output, (new_cache_k, new_cache_v)

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.int32,
        training: bool = None,
    ):
        """

        N: number of chunks
        R: number of right contexts in one chunk
        T: Total_R + Tmax

        Args:
            x (tf.Tensor): [B, T, D]
            mask (tf.int32): [B, 1, T, T]

        Returns:
            tf.Tensor: [B, T, D]
        """
        # 1. perform layer norm
        x = self.ln_in(inputs)

        # 2. calculate q k,v for all timesteps -> [B, T, H, D/H]
        q = tf.einsum("...TI,HIO->...THO", x, self.query_kernel)
        k = tf.einsum("...TI,HIO->...THO", x, self.key_kernel)
        v = tf.einsum("...TI,HIO->...THO", x, self.value_kernel)

        x = self.attend(inputs, q, k, v, mask, training)

        return x
