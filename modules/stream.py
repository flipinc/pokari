from abc import ABC

import abstractmethod
import torch

from modules.emformer_encoder import EmformerEncoder

# TODO: if this module gets big enough, this should be managed by hydra


def get_stream(preprocessor, encoder, decoder):
    if isinstance(encoder, EmformerEncoder):
        return EmformerStream(preprocessor, encoder, decoder)
    else:
        raise NotImplementedError("Streaming for this encoder is not yet implemented.")


class Stream(ABC):
    @abstractmethod
    def stream(self, audio_signals, audio_lens, cache):
        pass


class EmformerStream(Stream):
    def __init__(self, preprocessor, encoder, decoder):
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder

    def stream(self, audio_signals, audio_lens, cache):
        if audio_lens is None:
            bs = audio_signals.size(0)
            # assuming all audio_signal has segment length (chunk length + right length)
            base_length = self.preprocessor.hop_length * self.encoder.subsampling_factor

            chunk_length = base_length * self.encoder.chunk_length
            right_length = base_length * self.encoder.right_length

            audio_lens = torch.Tensor([chunk_length + right_length] * bs).to(
                audio_signals.device
            )

        if not cache:
            cache_k = cache_v = cache_rnn_state = None
        else:
            cache_k, cache_v, cache_rnn_state = cache

        audio_signals, audio_lens = self.preprocessor(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
        )

        encoded_signals, encoded_lens, cache_k, cache_v = self.encoder.stream(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
            cache_k=cache_k,
            cache_v=cache_v,
        )

        current_hypotheses, cache_rnn_state = self.decoder.generate_hypotheses(
            encoded_signals,
            encoded_lens,
            cache_rnn_state,
            mode="stream",
        )

        return current_hypotheses, (cache_k, cache_v, cache_rnn_state)
