from typing import Callable, List, Optional, Union

import torch
from frontends.audio_preprocess import WaveformFeaturizer
from frontends.text_parser import make_parser

from datasets.manifest import ManifestCollector


class AudioToCharDataset(torch.utils.data.Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
    """

    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor=None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,  # silenceをtrimするかどうか
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        load_audio: bool = True,
        parser: Union[str, Callable] = "en",
        add_misc: bool = False,
    ):
        self.labels = labels
        self.parser = make_parser(
            labels=labels,
            name=parser,
            unk_id=unk_index,
            blank_id=blank_index,
            do_normalize=normalize,
        )

        self.collection = ManifestCollector(
            manifests_files=manifest_filepath.split(","),
            parser=self.parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=augmentor
        )
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.load_audio = load_audio
        self._add_misc = add_misc

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file,
                offset=offset,
                duration=sample.duration,
                trim=self.trim,
                orig_sr=sample.orig_sr,
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t, tl = sample.text_tokens, len(sample.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        if self._add_misc:
            misc = dict()
            misc["id"] = sample.id
            misc["text_raw"] = sample.text_raw
            misc["speaker"] = sample.speaker
            output = (output, misc)

        return output

    def __len__(self):
        return len(self.collection)

    def collate_fn(self, batch):
        """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                  LongTensor):  A tuple of tuples of signal, signal lengths,
                  encoded tokens, and encoded tokens length.  This collate func
                  assumes the signals are 1d torch tensors (i.e. mono audio).
        """
        _, audio_lengths, _, tokens_lengths = zip(*batch)
        max_audio_len = 0
        has_audio = audio_lengths[0] is not None
        if has_audio:
            max_audio_len = max(audio_lengths).item()
        max_tokens_len = max(tokens_lengths).item()

        audio_signal, tokens = [], []
        for sig, sig_len, tokens_i, tokens_i_len in batch:
            if has_audio:
                sig_len = sig_len.item()
                if sig_len < max_audio_len:
                    pad = (0, max_audio_len - sig_len)
                    sig = torch.nn.functional.pad(sig, pad)
                audio_signal.append(sig)
            tokens_i_len = tokens_i_len.item()
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=self.pad_id)
            tokens.append(tokens_i)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.stack(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)

        return audio_signal, audio_lengths, tokens, tokens_lengths
