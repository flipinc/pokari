import math

import numpy as np
from modules.subsample import StackSubsample, VggSubsample

import tensorflow as tf


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
                feat_in=feat_out,
                feat_out=dim_model,
                conv_channels=subsampling_dim,
                activation=tf.nn.relu,
                name=f"{self.name}_vgg_subsample",
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
                name=f"{self.name}_{i}_block",
            )
            self.layers.append(layer)

    def punch_out_ones(
        self,
        bs,
        height,
        width,
        top,
        bottom,
        left,
        right,
    ):
        """Punch out (fill in ones) from size [Batch, height, width] mask

        Args:
            bs: batch size
            height: height of mask
            width: width of mask
            top: begining of y index which must be filled in as one
            bottom: end of y index which must be filled in as one
            left: begining of x index which must be filled in as one
            right: end of x index which must be filled in as one
        Returns:
            tf.int32: punched out mask

        TODO: there is another way to implement this. Instead of scatter_nd,
        sandwitch zero-tensor on one-tensor.

        """
        row_indices = tf.TensorArray(tf.int32, size=right - left, clear_after_read=True)
        counter = tf.constant(0, tf.int32)
        for i in tf.range(left, right):
            row_indices = row_indices.write(counter, tf.expand_dims(i, axis=0))
            counter += tf.constant(1, tf.int32)
        row_indices = row_indices.stack()

        value = tf.ones([right - left], tf.int32)

        row = tf.scatter_nd(row_indices, value, [width])
        row = tf.expand_dims(row, axis=0)  # [1, width]

        belt = tf.tile(row, [bottom - top, 1])  # [bottom - top, width]

        col_indices = tf.TensorArray(tf.int32, size=bottom - top, clear_after_read=True)
        counter = tf.constant(0, tf.int32)
        for i in tf.range(top, bottom):
            col_indices = col_indices.write(counter, tf.expand_dims(i, axis=0))
            counter += tf.constant(1, tf.int32)
        col_indices = col_indices.stack()

        target = tf.scatter_nd(col_indices, belt, [height, width])
        target = tf.expand_dims(target, axis=0)  # [1, height, width]

        targets = tf.tile(target, [bs, 1, 1])

        return targets

    def pad_mask(self, vertical_min, vertical_max, horizontal_min, horizontal_max):
        """Pad emformer mask
        For efficienct, mask is divided into three areas as follows:

        0 0 0 0 1 1
        0 0 0 0 1 1
        0 0 0 0 1 1
        2 2 2 2 2 2
        2 2 2 2 2 2

        Each number represents different mask.
        """
        mask_top_left = tf.ones([vertical_min, horizontal_min], tf.int32)
        mask_top_right = tf.zeros(
            [vertical_min, horizontal_max - horizontal_min], tf.int32
        )
        mask_bottom = tf.zeros([vertical_max - vertical_min, horizontal_max], tf.int32)
        mask_pad_mask = tf.concat([mask_top_left, mask_top_right], axis=1)
        mask_pad_mask = tf.concat([mask_pad_mask, mask_bottom], axis=0)

        return mask_pad_mask

    def create_mask(self, audio_lens: tf.int32, t_max: tf.int32):
        """Tensorflow implementation of emformer attention mask

        There are four types of masks.
        - mask_body: A attention mask for every timestep.
        - mask_left: A attention mask for every timestep. Hard copied
            right contexts included.
        - mark_right: A attention mask for hard copied right contexts.
        - mark_diagnal: A attention mask for hard copied right contexts
            with copied right contexts.

        Note: Only used during training.
        Note: In graph mode this is as fast as numpy impl. But using this in eager mode
        is not recommended

        Args:
            audio_lens (tf.Tensor): [B]
            t_max (int): this cannot be inferred by audio_lens because longest input_len
                may be padded for efficiency
        """
        bs = tf.shape(audio_lens)[0]
        num_chunks = tf.cast(tf.math.ceil(t_max / self.chunk_length), tf.int32)

        # TODO: this is allocating more than what is actually required
        upperbound_right_length = (num_chunks - 1) * self.right_length

        mask_body = tf.zeros([bs, t_max, t_max], tf.int32)
        mask_left = tf.zeros([bs, t_max, upperbound_right_length], tf.int32)
        mask_diagnal = tf.zeros(
            [bs, upperbound_right_length, upperbound_right_length], tf.int32
        )
        mask_right = tf.zeros([bs, upperbound_right_length, t_max], tf.int32)

        right_indexes = tf.TensorArray(
            tf.int32, size=0, dynamic_size=True, clear_after_read=True
        )
        for i in tf.range(num_chunks):
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

            mask_body += self.punch_out_ones(
                bs,
                width=t_max,
                height=t_max,
                left=left_offset,
                right=end_offset,
                top=start_offset,
                bottom=end_offset,
            )

            # last segment does not have right context
            if i == (num_chunks - 1):
                break

            # 2. hard copy right contexts
            remaining_space = t_max - (end_offset + self.right_length)
            if remaining_space >= 0:
                this_right_length = self.right_length
            else:
                this_right_length = t_max - end_offset

            mask_left += self.punch_out_ones(
                bs,
                height=t_max,
                width=upperbound_right_length,
                top=start_offset,
                bottom=start_offset + self.chunk_length,
                left=right_indexes.size(),
                right=right_indexes.size() + this_right_length,
            )

            mask_diagnal += self.punch_out_ones(
                bs,
                height=upperbound_right_length,
                width=upperbound_right_length,
                top=right_indexes.size(),
                bottom=right_indexes.size() + this_right_length,
                left=right_indexes.size(),
                right=right_indexes.size() + this_right_length,
            )

            mask_right += self.punch_out_ones(
                bs,
                height=upperbound_right_length,
                width=t_max,
                top=right_indexes.size(),
                bottom=right_indexes.size() + this_right_length,
                left=left_offset,
                right=end_offset,
            )

            for offset in tf.range(
                end_offset, end_offset + this_right_length, dtype=tf.int32
            ):
                right_indexes = right_indexes.write(right_indexes.size(), offset)

        right_indexes = right_indexes.stack()
        right_size = tf.shape(right_indexes)[0]

        # 3. remove unused right contexts
        kept_indicies = tf.range(right_size, dtype=tf.int32)
        mask_left = tf.gather(mask_left, kept_indicies, axis=2)
        mask_diagnal = tf.gather(mask_diagnal, kept_indicies, axis=1)
        mask_diagnal = tf.gather(mask_diagnal, kept_indicies, axis=2)
        mask_right = tf.gather(mask_right, kept_indicies, axis=1)

        # 4. mask paddings
        # TODO: there should be a way to parallelize this
        padded_mask_body = tf.TensorArray(tf.int32, size=bs, clear_after_read=True)
        padded_mask_right = tf.TensorArray(tf.int32, size=bs, clear_after_read=True)
        padded_mask_left = tf.TensorArray(tf.int32, size=bs, clear_after_read=True)
        padded_mask_diagnal = tf.TensorArray(tf.int32, size=bs, clear_after_read=True)
        for i in tf.range(bs):
            max_len = audio_lens[i]

            # [None, 1]
            to_be_padded_right_indicies = tf.where(right_indexes >= max_len)
            if tf.shape(to_be_padded_right_indicies)[0] > 0:
                # get raw value
                pad_begin_index = tf.cast(to_be_padded_right_indicies[0][0], tf.int32)
            else:
                # if preprocessing are implemented right, there shouldn't be
                # audio features that are less than the chunk size.
                # But for now, temporary value is used for numerical efficiency.
                # TODO: Does this affect any code?.
                pad_begin_index = right_size

            # 4.1 pad mask_body
            mask_body_pad_mask = self.pad_mask(
                vertical_min=max_len,
                vertical_max=t_max,
                horizontal_min=max_len,
                horizontal_max=t_max,
            )
            padded_mask_body = padded_mask_body.write(
                i, mask_body[i] * mask_body_pad_mask
            )

            # 4.2 pad mask_right
            mask_right_pad_mask = self.pad_mask(
                vertical_min=pad_begin_index,
                vertical_max=right_size,
                horizontal_min=max_len,
                horizontal_max=t_max,
            )
            padded_mask_right = padded_mask_right.write(
                i, mask_right[i] * mask_right_pad_mask
            )

            # 4.3 pad mask_left
            mask_left_pad_mask = self.pad_mask(
                vertical_min=max_len,
                vertical_max=t_max,
                horizontal_min=pad_begin_index,
                horizontal_max=right_size,
            )
            padded_mask_left = padded_mask_left.write(
                i, mask_left[i] * mask_left_pad_mask
            )

            # 4.4 pad mask_diagnal
            mask_diagnal_pad_mask = self.pad_mask(
                vertical_min=pad_begin_index,
                vertical_max=right_size,
                horizontal_min=pad_begin_index,
                horizontal_max=right_size,
            )
            padded_mask_diagnal = padded_mask_diagnal.write(
                i, mask_diagnal[i] * mask_diagnal_pad_mask
            )

        # 5. stack all masks
        mask_body = padded_mask_body.stack()
        mask_right = padded_mask_right.stack()
        mask_left = padded_mask_left.stack()
        mask_diagnal = padded_mask_diagnal.stack()

        # 6. concatenate all masks
        mask_top = tf.concat([mask_diagnal, mask_right], axis=-1)
        mask_bottom = tf.concat([mask_left, mask_body], axis=-1)
        mask = tf.concat([mask_top, mask_bottom], axis=-2)
        mask = tf.expand_dims(mask, axis=1)

        # return tf.equal(mask, 1), right_indexes

        return tf.cast(mask, tf.float32), right_indexes

    def np_create_mask(self, audio_lens: np.array, t_max: int):
        """Numpy implementatino of emformer attention mask"""
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

        # return mask == 1, right_indexes

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

        # 3. loop over layers while saving cache at the same time
        new_cache_k = tf.TensorArray(
            self.dtype,
            size=len(self.layers),
            clear_after_read=True,
            element_shape=tf.TensorShape(
                (
                    None,  # batch size
                    self.left_length,
                    self.num_heads,
                    self.dim_model // self.num_heads,
                )
            ),
        )
        new_cache_v = tf.TensorArray(
            self.dtype,
            size=len(self.layers),
            clear_after_read=True,
            element_shape=tf.TensorShape(
                (
                    None,  # batch size
                    self.left_length,
                    self.num_heads,
                    self.dim_model // self.num_heads,
                )
            ),
        )
        for i, layer in enumerate(self.layers):
            x, (temp_cache_k, temp_cache_v) = layer.stream(x, cache[0][i], cache[1][i])
            new_cache_k = new_cache_k.write(i, temp_cache_k)
            new_cache_v = new_cache_v.write(i, temp_cache_v)
        new_cache_k = new_cache_k.stack()
        new_cache_v = new_cache_v.stack()
        new_cache = tf.stack([new_cache_k, new_cache_v], axis=0)

        # 4. Trim right context
        x = x[:, : self.chunk_length, :]

        return x, new_cache

    def get_initial_state(self, batch_size: int):
        """Get initial state for streaming emformer"""
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

    def call(self, audio_features: tf.Tensor, audio_lens: tf.int32):
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
        # mask, right_indexes = self.create_mask(audio_lens, t_new)
        mask, right_indexes = tf.numpy_function(
            self.np_create_mask, [audio_lens, t_new], [tf.float32, tf.int32]
        )

        # 4. Hard copy right context and prepare input for the first iteration
        # [B, Total_R+Tmax, D]
        x_right = tf.gather(x, right_indexes, axis=1)
        x = tf.concat([x_right, x], axis=1)

        # 5. loop over layers.
        for layer in self.layers:
            x = layer(x, mask)

        # 6. Trim copied right context
        x = x[:, len(right_indexes) :, :]

        # tf.print("🐳", x)

        return x, audio_lens


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
        name: str = "emformer_block",
    ):
        super().__init__(name=name)

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

        self.softmax = tf.keras.layers.Softmax()

        self.attn_dropout = tf.keras.layers.Dropout(dropout_attn)

        self.linear_out_1 = tf.keras.layers.Dense(dim_ffn, activation="relu")
        self.linear_out_2 = tf.keras.layers.Dense(dim_model)

    def attend(
        self,
        x: tf.Tensor,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: tf.Tensor = None,
    ):
        bs = tf.shape(q)[0]

        # 1. get attention scores -> [B, H, Total_R+Tmax, Total_R+Tmax]
        attn_scores = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.d_k)

        # 2. apply mask and softmax
        if mask is not None:
            if self.dtype == tf.float16:
                # for amp
                inf_neg = tf.float16.min
            else:
                # using float(-inf) or -math.inf gives nan after softmax
                inf_neg = -1e9

            attn_scores += inf_neg * (1.0 - mask)
            attn_probs = tf.nn.softmax(attn_scores, axis=-1)
            # attn_probs = attn_probs * mask
        else:
            attn_probs = tf.nn.softmax(attn_scores, axis=-1)

        attn_probs = self.attn_dropout(attn_probs)

        # 3. attend and add residual
        output = tf.matmul(attn_probs, v)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (bs, -1, self.num_heads * self.d_k))
        output += x
        attn_out = self.ln_out_1(output)

        # 5. FFN modules
        output = self.linear_out_1(attn_out)
        output = self.linear_out_2(output)
        # emformer impl is slightly different. before ln_out_1 is added in emformer
        output += attn_out
        output = self.ln_out_2(output)

        # ref: https://github.com/Kyubyong/transformer/blob/master/modules.py

        return output

    def stream(
        self,
        x: tf.Tensor,
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
            x (tf.Tensor): [B, C+R, D]
            cache_k (tf.Tensor): [N, B, H, L, D/H]
            cache_v (tf.Tensor): [N, B, H, L, D/H]
        """
        bs = tf.shape(x)[0]

        # 1. apply layer norm
        x = self.ln_in(x)

        # 2. calculate q -> [B, H, C+R, D]
        q = tf.reshape(self.linear_q(x), (bs, -1, self.num_heads, self.d_k))
        q = tf.transpose(q, (0, 2, 1, 3))

        # 3. calculate k and v -> [B, H, L+C+R, D]
        k_cr = tf.reshape(self.linear_k(x), (bs, -1, self.num_heads, self.d_k))
        # we need to include previous cache as left length can extend beyond chunk.
        k = tf.concat([cache_k, k_cr], axis=1)
        new_cache_k = k[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]
        k = tf.transpose(k, (0, 2, 1, 3))

        v_cr = tf.reshape(self.linear_v(x), (bs, -1, self.num_heads, self.d_k))
        v = tf.concat([cache_v, v_cr], axis=1)
        new_cache_v = v[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]
        v = tf.transpose(v, (0, 2, 1, 3))

        output = self.attend(x, q, k, v)

        return output, (new_cache_k, new_cache_v)

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.int32,
    ):
        """

        N: number of chunks
        R: number of right contexts in one chunk

        Args:
            x (tf.Tensor): [B, Total_R+Tmax, D]
            mask (tf.int32): [B, 1, Total_R+Tmax, Total_R+Tmax]

        Returns:
            tf.Tensor: [B, Total_R+Tmax, D]
        """
        bs = tf.shape(inputs)[0]

        # 1. perform layer norm
        x = self.ln_in(inputs)

        # 2. calculate q k,v for all timesteps -> [B, H, Total_R+Tmax, D/H]
        q = tf.reshape(self.linear_q(x), (bs, -1, self.num_heads, self.d_k))
        k = tf.reshape(self.linear_k(x), (bs, -1, self.num_heads, self.d_k))
        v = tf.reshape(self.linear_v(x), (bs, -1, self.num_heads, self.d_k))
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        x = self.attend(inputs, q, k, v, mask)

        return x
