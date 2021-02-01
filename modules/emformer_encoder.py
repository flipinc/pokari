import math

import numpy as np
import torch
import torch.nn as nn

from modules.subsample import VggSubsample


class EmformerEncoder(nn.Module):
    def __init__(
        self, cfg,
    ):
        super().__init__()

        self.subsampling = "vgg"
        self.stack_length = cfg.stack_length
        self.num_layers = cfg.num_layers
        self.left_length = cfg.left_length
        self.chunk_length = cfg.chunk_length
        self.right_length = cfg.right_length

        feat_in = cfg.feat_in
        feat_out = cfg.feat_out
        num_heads = cfg.num_heads
        dim_model = cfg.dim_model
        dim_ffn = cfg.dim_ffn
        dropout_attn = cfg.dropout_attn

        self.linear = nn.Linear(feat_in, feat_out)
        self.subsample = VggSubsample(
            subsampling_factor=self.stack_length,
            feat_in=feat_out,
            feat_out=dim_model,
            conv_channels=256,
            activation=nn.ReLU(),
        )

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = EmformerBlock(
                num_heads=num_heads,
                dim_model=dim_model,
                dim_ffn=dim_ffn,
                dropout_attn=dropout_attn,
                left_length=self.left_length,
                chunk_length=self.chunk_length,
                right_length=self.right_length,
            )
            self.layers.append(layer)

    def create_mask(self, input_lens, t_max, device):
        """Emformer attention mask

        There are four types of masks.
        - mask_body: A mask that's used for transformer as normal
        - mask_left: A mask for hard copied inputs that's used for right contexts.
        - mark_right: Same as mask_left. It is just a transposed vector of mask_left.
        - mark_diagnal: A diagnal mask used for hard copied right contexts.

        Note: Only used during training.

        Args:
            input_lens (torch.Tensor): [B]
            t_max (int): this cannot be inferred by input_lens because longest input_len
                may be padded for efficiency
            device (torch.device)
        """
        bs = input_lens.size(0)
        num_chunks = math.ceil(t_max / self.chunk_length)

        mask_body = torch.zeros(bs, t_max, t_max)
        # TODO: this is allocating more than what is actually required
        mask_left = torch.zeros(bs, t_max, (num_chunks - 1) * self.right_length)

        right_indexes = []
        total_right_length = 0
        for i in range(num_chunks):
            # 1. enable diagnal and left chunks
            left_index = (
                self.chunk_length * i - self.left_length
                if (self.chunk_length * i - self.left_length) >= 0
                else 0
            )
            chunk_start = self.chunk_length * i
            chunk_end = (
                self.chunk_length * (i + 1)
                if self.chunk_length * (i + 1) <= t_max
                else t_max
            )
            mask_body[:, chunk_start:chunk_end, left_index:chunk_end] = 1

            # last segment does not have right context
            if i == (num_chunks - 1):
                break

            # 2. hard copy right contexts
            remaining_space = t_max - (chunk_end + self.right_length)
            if remaining_space >= 0:
                this_right_length = self.right_length
            elif remaining_space < 0:
                this_right_length = t_max - chunk_end

            right_indexes.extend(range(chunk_end, chunk_end + this_right_length))

            mask_left[
                :,
                chunk_start : chunk_start + self.chunk_length,
                i * self.right_length : i * self.right_length + this_right_length,
            ] = 1
            total_right_length += this_right_length

        mask_left = mask_left[:, :, :total_right_length]

        # 3. create a diagnal mask without padding
        right_size = len(right_indexes)
        mask_diagnal = torch.diag(torch.ones(right_size))
        mask_diagnal = mask_diagnal.expand(1, right_size, right_size)
        mask_diagnal = mask_diagnal.repeat(bs, 1, 1)

        # 4. mask paddings
        # TODO: there should be a way to parallelize this
        for i in range(bs):
            max_len = int(input_lens[i])

            # 4.1 pad mask_body
            mask_body[i, :, max_len:] = 0
            mask_body[i, max_len:, :] = 0

            to_be_padded = torch.nonzero(torch.Tensor(right_indexes) >= max_len)
            if to_be_padded.size(0) > 0:
                pad_begin_index = int(to_be_padded[0])

                # 4.2 pad mask_left
                mask_left[i, :, pad_begin_index:] = 0

                # 4.3 pad mask_diagnal
                mask_diagnal[i, :, pad_begin_index:] = 0

        mask_right = mask_left.transpose(1, 2)
        mask_top = torch.cat([mask_diagnal, mask_right], dim=-1)

        mask_bottom = torch.cat([mask_left, mask_body], dim=-1)
        mask = torch.cat([mask_top, mask_bottom], dim=-2)
        mask = mask.unsqueeze(1)

        return (mask == 0).to(device), right_indexes

    def recognize(
        self, audio_signal, length, cache_q=None, cache_v=None, cache_audio=None
    ):
        # 1. projection
        audio_signal = audio_signal.transpose(1, 2)
        audio_signal = self.linear(audio_signal)

        # 2. vgg subsampling
        if self.subsampling == "stack":
            bs, t_max, idim = audio_signal.shape
            t_new = math.ceil(t_max / self.stack_length)
            audio_signal = audio_signal.contiguous().view(
                bs, t_new, idim * self.stack_length
            )
            length = torch.ceil(length / self.stack_length).int()
        elif self.subsampling == "vgg":
            cache = audio_signal[:, -self.subsample.left_padding :, :]
            audio_signal, length = self.subsample(audio_signal, length, cache_audio)
            cache_audio = cache
            del cache

        # 3. loop over layers while saving cache
        if not cache_q:
            num_layers = len(self.layers)
            cache_q = cache_v = [None] * num_layers

        output = audio_signal
        for i, layer in enumerate(self.layers):
            output, (cache_q[i], cache_v[i]) = layer.recognize(
                output, cache_q[i], cache_v[i]
            )

        # 4. Trim right context
        output = output[:, : self.chunk_length, :].transpose(1, 2)
        length = torch.IntTensor([output.size(2)])

        return output, length, cache_q, cache_v, cache_audio

    def forward(self, audio_signal, length):
        """

        D_1: number of mels. This number has to match D_2 after frame stacking
        D_2: encoder dim.

        audio_signal: [B, D_1, Tmax]
        length: [B]

        output: [B, D_2, Tmax]

        """

        # 1. projection
        audio_signal = audio_signal.transpose(1, 2)
        audio_signal = self.linear(audio_signal)

        # 2. subsampling
        if self.subsampling == "stack":
            bs, t_max, idim = audio_signal.shape
            t_new = math.ceil(t_max / self.stack_length)
            audio_signal = audio_signal.contiguous().view(
                bs, t_new, idim * self.stack_length
            )
            length = torch.ceil(length / self.stack_length).int()
        elif self.subsampling == "vgg":
            audio_signal, length = self.subsample(audio_signal, length)
            t_new = audio_signal.size(1)

        # 3. create attention mask
        mask, right_indexes = self.create_mask(length, t_new, audio_signal.device)

        # 4. Hard copy right context and prepare input for the first iteration
        # [B, Total_R+Tmax, D]
        output = torch.cat([audio_signal[:, right_indexes, :], audio_signal], dim=1)

        # 5. loop over layers.
        for layer in self.layers:
            output = layer(output, mask)

        # 6. Trim copied right context
        output = output[:, len(right_indexes) :, :].transpose(1, 2)

        return output, length


class EmformerBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        dim_model,
        dim_ffn,
        dropout_attn,
        left_length,
        chunk_length,
        right_length,
    ):
        super().__init__()

        self.left_length = left_length
        self.chunk_length = chunk_length
        self.right_length = right_length

        self.d_k = dim_model // num_heads
        self.num_heads = num_heads

        self.ln_in = nn.LayerNorm(dim_model)
        self.ln_out_1 = nn.LayerNorm(dim_model)
        self.ln_out_2 = nn.LayerNorm(dim_model)

        self.linear_q = nn.Linear(dim_model, dim_model)
        self.linear_k = nn.Linear(dim_model, dim_model)
        self.linear_v = nn.Linear(dim_model, dim_model)

        self.attn_dropout = nn.Dropout(dropout_attn)

        self.linear_out_1 = nn.Linear(dim_model, dim_ffn)
        self.linear_out_2 = nn.Linear(dim_ffn, dim_model)

    def recognize(self, input, cache_k=None, cache_v=None):
        """
        At inference time, the notion of Left, Chunk, Right context still exists,
        because without them, compatibility with training will be lost.
        For example, at training time, the output for t_0, t_1, t_2 is determined
        by the input of t_0, t_1, t_2 PLUS right contexts.
        This is also true for left contexts.

        Args:
            input (torch.Tensor): [1, C+R, D]
            cache_k (torch.Tensor): [1, H, L, D/H]
            cache_v (torch.Tensor): [1, H, L, D/H]
        """
        input = self.ln_in(input)

        # 1. calculate q -> [B, H, C+R, D]
        q = self.linear_q(input).view(1, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)

        # 2. calculate k and v -> [B, H, L+C+R, D]
        k_cr = self.linear_k(input).view(1, -1, self.num_heads, self.d_k)
        k = k_cr if cache_k is None else torch.cat([cache_k, k_cr], dim=1)
        # we need to include previous cache as left length can extend beyond chunk.
        cache_k = k[
            :, -(self.left_length + self.right_length) : self.right_length, :, :
        ]
        k = k.transpose(1, 2)

        v_cr = self.linear_v(input).view(1, -1, self.num_heads, self.d_k)
        v = v_cr if cache_v is None else torch.cat([cache_v, v_cr], dim=1)
        cache_v = v[
            :, -(self.left_length + self.right_length) : self.right_length, :, :
        ]
        v = v.transpose(1, 2)

        # 3. get attention score -> [B, H, C+R, L+C+R]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. softmax, and get attention probability
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 5. attend and add residual -> [B, H, C+R, D]
        output = torch.matmul(attn_probs, v)
        output = (
            output.transpose(1, 2).contiguous().view(1, -1, self.num_heads * self.d_k)
        )
        attn_out = output + input

        output = self.ln_out_1(attn_out)

        # 6. feed forward and add residual
        output = self.linear_out_1(output)
        output = self.linear_out_2(output) + attn_out

        output = self.ln_out_2(output)

        return output, (cache_k, cache_v)

    def forward(self, input, mask):
        """

        N: number of chunks
        R: number of right contexts in one chunk

        input: [B, Total_R+Tmax, D]
        mask: [B, 1, Total_R+Tmax, Total_R+Tmax]

        output: [B, Total_R+Tmax, D]

        """
        bs, t_max, _ = input.shape

        # 1. perform layer norm
        input = self.ln_in(input)

        # 2. calculate q k,v for all timesteps -> [B, H, Total_R+Tmax, D/H]
        q = self.linear_q(input).view(bs, -1, self.num_heads, self.d_k)
        k = self.linear_k(input).view(bs, -1, self.num_heads, self.d_k)
        v = self.linear_v(input).view(bs, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. get attention scores -> [B, H, Total_R+Tmax, Total_R+Tmax]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. apply mask and softmax
        dtype = np.float16 if attn_scores.dtype == torch.float16 else np.float32
        attn_scores = attn_scores.masked_fill(mask, np.finfo(dtype).min)
        attn_probs = torch.softmax(attn_scores, dim=-1).masked_fill(mask, 0.0)
        attn_probs = self.attn_dropout(attn_probs)

        # 5. attend and add residual
        # NEXT: check the correct implementation. especially around residual
        output = torch.matmul(attn_probs, v)
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        )
        attn_out = output + input

        # 6. layer norm
        output = self.ln_out_1(attn_out)

        # 7. feed forward and add residual
        output = self.linear_out_1(output)
        output = self.linear_out_2(output) + attn_out

        # 8. layer norm
        output = self.ln_out_2(output)

        return output
