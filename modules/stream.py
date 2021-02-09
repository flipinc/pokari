import torch


class EmformerStream(object):
    def __init__(self, preprocessor, encoder, decoder):
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, audio_signals, audio_lens, cache=None):
        if audio_lens is None:
            bs = audio_signals.size(0)
            # assuming all audio_signal has segment length (chunk length + right length)
            base_length = self.preprocessor.hop_length * self.encoder.subsampling_factor

            chunk_length = base_length * self.encoder.chunk_length
            right_length = base_length * self.encoder.right_length

            audio_lens = torch.Tensor([chunk_length + right_length] * bs).to(
                audio_signals.device
            )

        if cache is None:
            cache_rnn_state = cache_k = cache_v = None
        else:
            cache_rnn_state, cache_k, cache_v = cache

        audio_signals, audio_lens = self.preprocessor(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
        )

        encoded_signals, encoded_lens, cache_k, cache_v = self.encoder(
            audio_signals=audio_signals,
            audio_lens=audio_lens,
            cache_k=cache_k,
            cache_v=cache_v,
            mode="stream",
        )

        current_hypotheses, cache_rnn_state = self.decoder(
            encoded_signals,
            encoded_lens,
            cache_rnn_state,
            mode="stream",
        )

        return current_hypotheses, (cache_rnn_state, cache_k, cache_v)
