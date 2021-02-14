import math
from typing import Optional

import torch
import torch.nn as nn

from modules.subsample import VggSubsample, stack_subsample


class EmformerEncoder(nn.Module):
    def __init__(
        self,
        subsampling: str,
        subsampling_factor: int,
        subsampling_dim: int,
        feat_in: int,
        num_layers: int,
        num_heads: int,
        dim_model: int,
        dim_ffn: int,
        dropout_attn: int,
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

        self.linear = nn.Linear(feat_in, feat_out)
        self.subsample = VggSubsample(
            subsampling_factor=self.subsampling_factor,
            feat_in=feat_out,
            feat_out=dim_model,
            conv_channels=subsampling_dim,
            activation=nn.ReLU(),
        )

        self.layers = nn.ModuleList()
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
        self, audio_lens: torch.Tensor, segment_length: int, device: torch.device
    ):
        bs = audio_lens.size(0)
        mask = torch.zeros(bs, segment_length, segment_length + self.left_length).to(
            device
        )

        # TODO: if cache_k and cache_v is not given, there is no point of attending
        # to the left chunk
        # TODO: there should be a way to parallelize this
        for i in range(bs):
            max_len = int(audio_lens[i])
            mask[i, :, max_len + self.left_length :] = 0
            mask[i, max_len:, :] = 0

        mask = mask.unsqueeze(1)

        return mask == 0

    def create_mask(self, audio_lens: torch.Tensor, t_max: int, device: torch.device):
        """Emformer attention mask

        There are four types of masks.
        - mask_body: A mask that's used for transformer as normal
        - mask_left: A mask for hard copied inputs that's used for right contexts.
        - mark_right: Same as mask_left. It is just a transposed vector of mask_left.
        - mark_diagnal: A diagnal mask used for hard copied right contexts.

        Note: Only used during training.

        Args:
            audio_lens (torch.Tensor): [B]
            t_max (int): this cannot be inferred by audio_lens because longest input_len
                may be padded for efficiency
            device (torch.device)
        """
        bs = audio_lens.size(0)
        num_chunks = math.ceil(t_max / self.chunk_length)

        # TODO: this is allocating more than what is actually required
        upperbound_right_length = (num_chunks - 1) * self.right_length

        mask_body = torch.zeros(bs, t_max, t_max)
        mask_left = torch.zeros(bs, t_max, upperbound_right_length)
        mask_diagnal = torch.zeros(
            [bs, upperbound_right_length, upperbound_right_length]
        )
        mask_right = torch.zeros([bs, upperbound_right_length, t_max])

        right_indexes = torch.empty(0).to(dtype=torch.long)
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

            right_indexes = torch.cat(
                [
                    right_indexes,
                    torch.arange(end_offset, end_offset + this_right_length),
                ]
            )

        # 3. remove unused right masks
        right_size = len(right_indexes)
        mask_left = mask_left[:, :, :right_size]
        mask_diagnal = mask_diagnal[:, :right_size, :right_size]
        mask_right = mask_right[:, :right_size, :]

        # 4. mask paddings
        # TODO: there should be a way to parallelize this
        for i in range(bs):
            max_len = int(audio_lens[i])

            # 4.1 pad mask_body
            mask_body[i, :, max_len:] = 0
            mask_body[i, max_len:, :] = 0
            mask_right[i, :, max_len:] = 0

            to_be_padded = torch.nonzero(right_indexes >= max_len)
            if to_be_padded.size(0) > 0:
                pad_begin_index = int(to_be_padded[0])

                mask_right[i, pad_begin_index:, :] = 0
                mask_left[i, :, pad_begin_index:] = 0
                mask_diagnal[i, pad_begin_index:, :] = 0
                mask_diagnal[i, :, pad_begin_index:] = 0

        # 5. concatenate all masks
        mask_top = torch.cat([mask_diagnal, mask_right], dim=-1)
        mask_bottom = torch.cat([mask_left, mask_body], dim=-1)
        mask = torch.cat([mask_top, mask_bottom], dim=-2)
        mask = mask.unsqueeze(1)

        return (mask == 0).to(device), right_indexes

    def stream(
        self,
        audio_signals: torch.Tensor,
        audio_lens: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ):
        # TODO: check all audio_lens are equal if batch size is > 1
        bs = audio_signals.size(0)

        # 1. projection
        x = audio_signals.transpose(1, 2)
        x = self.linear(x)

        # 2. vgg subsampling
        if self.subsampling == "stack":
            x, audio_lens = stack_subsample(x, audio_lens, self.subsampling_factor)
        elif self.subsampling == "vgg":
            x, audio_lens = self.subsample(x, audio_lens)

        # 3. create padding mask
        segment_length = x.size(1)
        mask = self.create_stream_mask(audio_lens, segment_length, x.device)

        # 4. loop over layers while saving cache at the same time
        if cache_k is None or cache_v is None:
            bs = audio_signals.size(0)
            # alternatively, this can be a list of size [B, L, Head, Dim] x num_layer
            # but for simplicity, everything is packed in torch.tensor
            cache_k = cache_v = torch.zeros(
                self.num_layers,
                bs,
                self.left_length,
                self.num_heads,
                self.dim_model // self.num_heads,
            ).to(audio_signals.device)

        for i, layer in enumerate(self.layers):
            x, cache = layer(x, mask, cache_k[i], cache_v[i], mode="stream")
            # this is always true. this is for torchscript typing
            if cache is not None:
                (cache_k[i], cache_v[i]) = cache
            else:
                raise ValueError("cache value for should be returned.")

        # 5. Trim right context
        x = x[:, : self.chunk_length, :].transpose(1, 2)
        for i in range(bs):
            audio_lens[i] = x.size(2)

        return x, audio_lens, cache_k, cache_v

    def full_context(self, audio_signals: torch.Tensor, audio_lens: torch.Tensor):
        # 1. projection
        x = audio_signals.transpose(1, 2)
        x = self.linear(x)

        # 2. subsampling
        if self.subsampling == "stack":
            x, audio_lens = stack_subsample(
                audio_signals, audio_lens, self.subsampling_factor
            )
        elif self.subsampling == "vgg":
            x, audio_lens = self.subsample(x, audio_lens)

        # 3. create attention mask
        t_new = x.size(1)
        mask, right_indexes = self.create_mask(audio_lens, t_new, x.device)

        # 4. Hard copy right context and prepare input for the first iteration
        # [B, Total_R+Tmax, D]
        x = torch.cat([x[:, right_indexes, :], x], dim=1)

        # 5. loop over layers.
        for layer in self.layers:
            x, _ = layer(x, mask)

        # 6. Trim copied right context
        x = x[:, len(right_indexes) :, :].transpose(1, 2)

        # 7. Experiment: Zero out padded parts
        bs = x.size(0)
        mask = torch.ones(x.size()).to(device=x.device)
        for idx in range(bs):
            mask[idx, :, audio_lens[idx] :] = 0
        x = x * mask

        return x, audio_lens, None, None

    def forward(
        self,
        audio_signals: torch.Tensor,
        audio_lens: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        mode: str = "full_context",
    ):
        """

        D_1: number of mels. This number has to match D_2 after frame stacking
        D_2: encoder dim.

        Args:
            audio_signals (torch.Tensor): [B, D_1, Tmax]
            audio_lens (torch.Tensor): [B]

        Returns:
            torch.Tensor: [B, D_2, Tmax]
        """

        if mode == "full_context":
            return self.full_context(audio_signals, audio_lens)
        elif mode == "stream":
            return self.stream(audio_signals, audio_lens, cache_k, cache_v)
        else:
            raise ValueError(f"Invalid mode {mode}")


class EmformerBlock(nn.Module):
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

        self.ln_in = nn.LayerNorm(dim_model)
        self.ln_out_1 = nn.LayerNorm(dim_model)
        self.ln_out_2 = nn.LayerNorm(dim_model)

        self.linear_q = nn.Linear(dim_model, dim_model)
        self.linear_k = nn.Linear(dim_model, dim_model)
        self.linear_v = nn.Linear(dim_model, dim_model)

        self.attn_dropout = nn.Dropout(dropout_attn)

        self.linear_out_1 = nn.Linear(dim_model, dim_ffn)
        self.linear_out_2 = nn.Linear(dim_ffn, dim_model)

    def attend(
        self,
        input: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ):
        bs = q.size(0)

        # 1. get attention scores -> [B, H, Total_R+Tmax, Total_R+Tmax]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. apply mask and softmax
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1).masked_fill(mask, 0.0)
        attn_probs = self.attn_dropout(attn_probs)

        # 3. attend and add residual
        output = torch.matmul(attn_probs, v)
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        )
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
        input: torch.Tensor,
        mask: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """
        Note: At inference time, the notion of Left, Chunk, Right contexts still exists
            because without them, the compatibility with training will be lost.
            For example, at training time, the output for t_0, t_1, t_2 is determined
            by the input of t_0, t_1, t_2 PLUS right contexts.
            This must be the case for inference too.

        N: number of layers

        Args:
            input (torch.Tensor): [B, C+R, D]
            cache_k (torch.Tensor): [N, B, H, L, D/H]
            cache_v (torch.Tensor): [N, B, H, L, D/H]
        """
        bs = input.size(0)

        # 1. apply layer norm
        input = self.ln_in(input)

        # 2. calculate q -> [B, H, C+R, D]
        q = self.linear_q(input).view(bs, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)

        # 3. calculate k and v -> [B, H, L+C+R, D]
        k_cr = self.linear_k(input).view(bs, -1, self.num_heads, self.d_k)
        # we need to include previous cache as left length can extend beyond chunk.
        k = torch.cat([cache_k, k_cr], dim=1)
        cache_k = k[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]
        k = k.transpose(1, 2)

        v_cr = self.linear_v(input).view(bs, -1, self.num_heads, self.d_k)
        v = torch.cat([cache_v, v_cr], dim=1)
        cache_v = v[
            :, -(self.left_length + self.right_length) : -self.right_length, :, :
        ]
        v = v.transpose(1, 2)

        output = self.attend(input, q, k, v, mask)

        return output, (cache_k, cache_v)

    def full_context(self, input: torch.Tensor, mask: torch.Tensor):
        bs = input.size(0)

        # 1. perform layer norm
        input = self.ln_in(input)

        # 2. calculate q k,v for all timesteps -> [B, H, Total_R+Tmax, D/H]
        q = self.linear_q(input).view(bs, -1, self.num_heads, self.d_k)
        k = self.linear_k(input).view(bs, -1, self.num_heads, self.d_k)
        v = self.linear_v(input).view(bs, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attend(input, q, k, v, mask)

        return output, None

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        mode: str = "full_context",
    ):
        """

        N: number of chunks
        R: number of right contexts in one chunk

        Args:
            input (torch.Tensor): [B, Total_R+Tmax, D]
            mask (torch.Tensor): [B, 1, Total_R+Tmax, Total_R+Tmax]

        Returns:
            torch.Tensor: [B, Total_R+Tmax, D]
        """

        if mode == "full_context":
            return self.full_context(input, mask)
        elif mode == "stream":
            if cache_k is None or cache_v is None:
                bs = input.size(0)
                cache_k = cache_v = torch.zeros(
                    bs,
                    self.left_length,
                    self.num_heads,
                    self.d_k,
                ).to(input.device)

            return self.stream(input, mask, cache_k, cache_v)
        else:
            raise ValueError(f"Invalid mode {mode}")
