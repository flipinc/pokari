import math
from typing import Optional

import numpy as np
import tensorflow as tf
from modules.subsample import VggSubsample, stack_subsample


class EmformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        feat_in: int,
        num_layers: int,
        num_heads: int,
        dim_model: int,
        dim_ffn: int,
        dropout_attn: int,
        subsampling: str,
        subsampling_factor: int,
        subsampling_dim: int,  # conv channels. used for vgg susampling
        left_length: int,
        chunk_length: int,
        right_length: int,
    ):
        super().__init__()

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
        self.subsample = VggSubsample(
            subsampling_factor=self.subsampling_factor,
            feat_in=feat_out,
            feat_out=dim_model,
            conv_channels=subsampling_dim,
            activation=tf.keras.activations.relu,
        )

        self.layers = []
        for i in range(self.num_layers):
            layer = EmformerBlock(
                num_heads=self.num_heads,
                dim_model=self.dim_model,
                dim_ffn=dim_ffn,
                dropout_attn=dropout_attn,
                left_length=self.left_length,
                chunk_length=self.chunk_length,
                right_length=self.right_length,
            )
            self.layers.append(layer)

    def create_stream_mask(
        self,
        audio_lens: np.array,
        segment_length: int,
    ):
        audio_lens = audio_lens.astype("int32")

        bs = audio_lens.shape[0]
        mask = np.zeros([bs, segment_length, segment_length + self.left_length])

        # TODO: if cache_k and cache_v is not given, there is no point of attending
        # to the left chunk
        # TODO: there should be a way to parallelize this
        for i in range(bs):
            max_len = int(audio_lens[i])
            mask[i, :, max_len + self.left_length :] = 0
            mask[i, max_len:, :] = 0

        mask = np.expand_dims(mask, axis=1)

        return mask == 0

    def create_mask(self, audio_lens: np.array, t_max: int):
        """Emformer attention mask

        There are four types of masks.
        - mask_body: A attention mask for every timestep.
        - mask_left: A attention mask for every timestep. Hard copied
            right contexts included.
        - mark_right: A attention mask for hard copied right contexts.
        - mark_diagnal: A attention mask for hard copied right contexts
            with copied right contexts.

        Note: Only used during training.

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
            mask_body[i, :, max_len:] = 0
            mask_body[i, max_len:, :] = 0
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

        return mask == 0, right_indexes

    def stream(
        self,
        audio_signals: tf.Tensor,
        audio_lens: tf.Tensor,
        cache_k: Optional[tf.Tensor] = None,
        cache_v: Optional[tf.Tensor] = None,
    ):
        bs = tf.shape(audio_signals)[0]

        if bs > 1:
            max_len = tf.math.reduce_max(audio_lens)
            min_len = tf.math.reduce_min(audio_lens)
            if max_len != min_len:
                raise ValueError(
                    "In streaming mode, all audio lens must be equal if batch size > 1"
                )

        # 1. projection
        x = tf.transpose(audio_signals, (0, 2, 1))
        x = self.linear(x)

        # 2. vgg subsampling
        if self.subsampling == "stack":
            x, audio_lens = stack_subsample(x, audio_lens, self.subsampling_factor)
        elif self.subsampling == "vgg":
            x, audio_lens = self.subsample(x, audio_lens)

        # 3. create padding mask
        segment_length = tf.shape(x)[1]
        mask = tf.numpy_function(
            self.create_stream_mask, [audio_lens, segment_length], [tf.bool]
        )

        # 4. loop over layers while saving cache at the same time
        if cache_k is None or cache_v is None:
            # alternatively, this can be a list of size [B, L, Head, Dim] x num_layer
            # but for simplicity, everything is packed in tf.Tensor
            cache_k = cache_v = tf.zeros(
                [
                    self.num_layers,
                    bs,
                    self.left_length,
                    self.num_heads,
                    self.dim_model // self.num_heads,
                ]
            )

        new_cache_k = tf.TensorArray(
            self.dtype, size=len(self.layers), clear_after_read=True
        )
        new_cache_v = tf.TensorArray(
            self.dtype, size=len(self.layers), clear_after_read=True
        )
        for i, layer in enumerate(self.layers):
            x, (temp_cache_k, temp_cache_v) = layer(
                x, mask, cache_k[i], cache_v[i], mode="stream"
            )
            new_cache_k = new_cache_k.write(i, temp_cache_k)
            new_cache_v = new_cache_v.write(i, temp_cache_v)
        new_cache_k = new_cache_k.stack()
        new_cache_v = new_cache_v.stack()

        # 5. Trim right context
        x = x[:, : self.chunk_length, :]
        x = tf.transpose(x, (0, 2, 1))

        new_audio_lens = tf.repeat(self.chunk_length, [bs])

        return x, new_audio_lens, new_cache_k, new_cache_v

    def full_context(self, audio_signals: tf.Tensor, audio_lens: tf.Tensor):
        # 1. projection
        x = tf.transpose(audio_signals, [0, 2, 1])
        x = self.linear(x)

        # 2. subsampling
        if self.subsampling == "stack":
            x, audio_lens = stack_subsample(x, audio_lens, self.subsampling_factor)
        elif self.subsampling == "vgg":
            x, audio_lens = self.subsample(x, audio_lens)

        # 3. create attention mask
        t_new = tf.shape(x)[1]
        mask, right_indexes = tf.numpy_function(
            self.create_mask, [audio_lens, t_new], [tf.bool, tf.int32]
        )

        # 4. Hard copy right context and prepare input for the first iteration
        # [B, Total_R+Tmax, D]
        x_right = tf.gather(x, right_indexes, axis=1)
        x = tf.concat([x_right, x], axis=1)

        # 5. loop over layers.
        for layer in self.layers:
            x, _ = layer(x, mask)

        # 6. Trim copied right context
        # TODO: does len() work in graph mode?
        x = x[:, len(right_indexes) :, :]
        x = tf.transpose(x, [0, 2, 1])

        return x, audio_lens, None, None

    def call(
        self,
        audio_signals: tf.Tensor,
        audio_lens: np.array,
        cache_k: Optional[tf.Tensor] = None,
        cache_v: Optional[tf.Tensor] = None,
        mode: str = "full_context",
    ):
        """

        D_1: number of mels. This number has to match D_2 after frame stacking
        D_2: encoder dim.

        Args:
            audio_signals (tf.Tensor): [B, D_1, Tmax]
            audio_lens (np.array): [B]

        Returns:
            tf.Tensor: [B, D_2, Tmax]
        """

        if mode == "full_context":
            return self.full_context(audio_signals, audio_lens)
        elif mode == "stream":
            return self.stream(audio_signals, audio_lens, cache_k, cache_v)
        else:
            raise ValueError(f"Invalid mode {mode}")


class EmformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        dim_model: int,
        dim_ffn: int,
        dropout_attn: int,
        left_length: int,
        chunk_length: int,
        right_length: int,
    ):
        super().__init__()

        self.left_length = left_length
        self.chunk_length = chunk_length
        self.right_length = right_length

        self.d_k = dim_model // num_heads
        self.num_heads = num_heads

        self.ln_in = tf.keras.layers.LayerNormalization()
        self.ln_out_1 = tf.keras.layers.LayerNormalization()
        self.ln_out_2 = tf.keras.layers.LayerNormalization()

        self.linear_q = tf.keras.layers.Dense(dim_model)
        self.linear_k = tf.keras.layers.Dense(dim_model)
        self.linear_v = tf.keras.layers.Dense(dim_model)

        self.attn_dropout = tf.keras.layers.Dropout(dropout_attn)

        self.linear_out_1 = tf.keras.layers.Dense(dim_ffn)
        self.linear_out_2 = tf.keras.layers.Dense(dim_model)

    def attend(
        self,
        input: tf.Tensor,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: tf.Tensor,
    ):
        bs = tf.shape(q)[0]

        # 1. get attention scores -> [B, H, Total_R+Tmax, Total_R+Tmax]
        attn_scores = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.d_k)

        # 2. apply mask and softmax
        attn_scores = tf.where(mask, attn_scores, float("-inf"))
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        attn_probs = tf.where(mask, attn_probs, 0.0)
        attn_probs = self.attn_dropout(attn_probs)

        # 3. attend and add residual
        output = tf.matmul(attn_probs, v)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (bs, -1, self.num_heads * self.d_k))
        attn_out = output + input

        # 4. layer norm
        output = self.ln_out_1(attn_out)

        # 5. feed forward and add residual
        output = self.linear_out_1(output)
        output = self.linear_out_2(output) + attn_out

        # 6. layer norm
        output = self.ln_out_2(output)

        return output

    def stream(
        self,
        input: tf.Tensor,
        mask: tf.Tensor,
        cache_k: tf.Tensor,
        cache_v: tf.Tensor,
    ):
        """
        Note: At inference time, the notion of Left, Chunk, Right contexts still exists
            because without them, the compatibility with training will be lost.
            For example, at training time, the output for t_0, t_1, t_2 is determined
            by the input of t_0, t_1, t_2 PLUS right contexts.
            This must be the case for inference too.

        N: number of layers

        Args:
            input (tf.Tensor): [B, C+R, D]
            cache_k (tf.Tensor): [N, B, H, L, D/H]
            cache_v (tf.Tensor): [N, B, H, L, D/H]
        """
        bs = tf.shape(input)[0]

        # 1. apply layer norm
        input = self.ln_in(input)

        # 2. calculate q -> [B, H, C+R, D]
        q = tf.reshape(self.linear_q(input), (bs, -1, self.num_heads, self.d_k))
        q = tf.transpose(q, (0, 2, 1, 3))

        # 3. calculate k and v -> [B, H, L+C+R, D]
        k_cr = tf.reshape(self.linear_k(input), (bs, -1, self.num_heads, self.d_k))
        # we need to include previous cache as left length can extend beyond chunk.
        k = tf.concat([cache_k, k_cr], axis=1)
        new_cache_k = k[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]
        k = tf.transpose(k, (0, 2, 1, 3))

        v_cr = tf.reshape(self.linear_v(input), (bs, -1, self.num_heads, self.d_k))
        v = tf.concat([cache_v, v_cr], axis=1)
        new_cache_v = v[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]
        v = tf.transpose(v, (0, 2, 1, 3))

        output = self.attend(input, q, k, v, mask)

        return output, (new_cache_k, new_cache_v)

    def full_context(self, input: tf.Tensor, mask: tf.Tensor):
        bs = tf.shape(input)[0]

        # 1. perform layer norm
        input = self.ln_in(input)

        # 2. calculate q k,v for all timesteps -> [B, H, Total_R+Tmax, D/H]
        q = tf.reshape(self.linear_q(input), (bs, -1, self.num_heads, self.d_k))
        k = tf.reshape(self.linear_k(input), (bs, -1, self.num_heads, self.d_k))
        v = tf.reshape(self.linear_v(input), (bs, -1, self.num_heads, self.d_k))
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        output = self.attend(input, q, k, v, mask)

        return output, None

    def call(
        self,
        input: tf.Tensor,
        mask: tf.Tensor,
        cache_k: Optional[tf.Tensor] = None,
        cache_v: Optional[tf.Tensor] = None,
        mode: str = "full_context",
    ):
        """

        N: number of chunks
        R: number of right contexts in one chunk

        Args:
            input (tf.Tensor): [B, Total_R+Tmax, D]
            mask (tf.Tensor): [B, 1, Total_R+Tmax, Total_R+Tmax]

        Returns:
            tf.Tensor: [B, Total_R+Tmax, D]
        """

        if mode == "full_context":
            return self.full_context(input, mask)
        elif mode == "stream":
            if cache_k is None or cache_v is None:
                bs = tf.shape(input)[0]
                cache_k = cache_v = tf.zeros(
                    [
                        bs,
                        self.left_length,
                        self.num_heads,
                        self.d_k,
                    ]
                )

            return self.stream(input, mask, cache_k, cache_v)
        else:
            raise ValueError(f"Invalid mode {mode}")
