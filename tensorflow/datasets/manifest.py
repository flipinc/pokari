import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from datasets.text_parser import CharParser


def collect_manifest(
    manifests_files: Union[str, List[str]],
    parser: CharParser,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    max_number: Optional[int] = None,
):
    ids, audio_files, durations, texts, offsets, speakers, orig_srs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for item in item_iter(manifests_files):
        ids.append(item["id"])
        audio_files.append(item["audio_file"])
        durations.append(item["duration"])
        texts.append(item["text"])
        offsets.append(item["offset"])
        speakers.append(item["speaker"])
        orig_srs.append(item["orig_sr"])

    data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0

    for id_, audio_file, duration, offset, text, speaker, orig_sr in zip(
        ids, audio_files, durations, offsets, texts, speakers, orig_srs
    ):
        # Duration filters.
        if min_duration is not None and duration < min_duration:
            duration_filtered += duration
            num_filtered += 1
            continue

        if max_duration is not None and duration > max_duration:
            duration_filtered += duration
            num_filtered += 1
            continue

        text_tokens = parser(text)
        if text_tokens is None:
            duration_filtered += duration
            num_filtered += 1
            continue

        total_duration += duration

        data.append(
            (
                id_,
                audio_file,
                duration,
                text_tokens,
                offset,
                text,
                speaker,
                orig_sr,
            )
        )

        # Max number of entities filter.
        if len(data) == max_number:
            break

    return data


def item_iter(
    manifests_files: Union[str, List[str]],
    parse_func: Callable[[str, Optional[str]], Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate through json lines of provided manifests.

    NeMo ASR pipelines often assume certain manifest files structure. In
    particular, each manifest file should consist of line-per-sample files with
    each line being correct json dict. Each such json dict should have a field
    for audio file string, a field for duration float and a field for text
    string. Offset also could be additional field and is set to None by
    default.

    Args:
        manifests_files: Either single string file or list of such -
            manifests to yield items from.

        parse_func: A callable function which accepts as input a single line
            of a manifest and optionally the manifest file itself,
            and parses it, returning a dictionary mapping from str -> Any.

    Yields:
        Parsed key to value item dicts.

    Raises:
        ValueError: If met invalid json line structure.
    """

    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]

    if parse_func is None:
        parse_func = parse_item

    k = -1
    for manifest_file in manifests_files:
        with open(os.path.expanduser(manifest_file), "r") as f:
            for line in f:
                k += 1
                item = parse_func(line, manifest_file)
                item["id"] = k

                yield item


def parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    item = json.loads(line)

    # Audio file
    if "audio_filename" in item:
        item["audio_file"] = item.pop("audio_filename")
    elif "audio_filepath" in item:
        item["audio_file"] = item.pop("audio_filepath")
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: "
            f"{line} without proper audio file key."
        )
    item["audio_file"] = os.path.expanduser(item["audio_file"])

    # Duration.
    if "duration" not in item:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: "
            f"{line} without proper duration key."
        )

    # Text.
    if "text" in item:
        pass
    elif "text_filepath" in item:
        with open(item.pop("text_filepath"), "r") as f:
            item["text"] = f.read().replace("\n", "")
    elif "normalized_text" in item:
        item["text"] = item["normalized_text"]
    else:
        raise ValueError(
            f"Manifest file {manifest_file} has invalid json line structure: "
            f"{line} without proper text key."
        )

    item = dict(
        audio_file=item["audio_file"],
        duration=item["duration"],
        text=item["text"],
        offset=item.get("offset", None),
        speaker=item.get("speaker", None),
        orig_sr=item.get("orig_sample_rate", None),
    )

    return item
