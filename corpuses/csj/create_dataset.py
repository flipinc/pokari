import csv
import os
import re

# some notes about csjconnect.pl
# - pend -> previous end
# - ostart -> original start
# - sgid -> segment id
# - spk_id -> speaker id
# - <s> -> start of sentence
# - <sp> -> separator
# - gap -> allowed maximum gap between previous and original segment


def create_dataset(
    path_to_train_words="/workspace/datasets/csj_train_word.tsv",
    path_to_val_words="/workspace/datasets/csj_val_word.tsv",
    out_dir="/workspace/datasets",
    max_duration=17,  # 17 is the length that can include every lines
    allowed_gap=0.5,
):
    """Create a transcript tsv file from all words created by `create_word.py`

    Number of characters/subwords to be used
    - CHR: 3084 -> https://arxiv.org/pdf/2102.07935.pdf
    - CHR: 3260 -> https://arxiv.org/pdf/2006.14941.pdf
    - CHR: 3262, 3500 -> http://www.interspeech2020.org/uploadfile/pdf/Mon-2-2-2.pdf
    - BPE: 7520 -> https://arxiv.org/pdf/2008.03822.pdf

    Writes datasets in following format:
    path\ttranscript\tduration\toffset

    """
    train_out_file = "csj_train.tsv"
    val_out_file = "csj_val.tsv"

    with open(os.path.join(out_dir, train_out_file), "w") as train_out:
        with open(os.path.join(out_dir, val_out_file), "w") as val_out:
            train_writer = csv.writer(train_out, delimiter="\t")
            val_writer = csv.writer(val_out, delimiter="\t")

            for idx, word_file in enumerate([path_to_train_words, path_to_val_words]):
                writer = val_writer if idx == 1 else train_writer

                with open(word_file, "r") as f:
                    previous_segment_id = -1
                    previous_path = ""
                    previous_end_offset = 0
                    transcript = ""

                    for line in f:
                        path, segment_id, word, start_offset, end_offset = line.split(
                            "\t"
                        )
                        start_offset = float(start_offset)
                        end_offset = float(end_offset)

                        if previous_segment_id == -1:
                            current_path = path
                            current_start_offset = start_offset
                            transcript = word
                        elif (
                            previous_segment_id == segment_id and previous_path == path
                        ):
                            transcript += word
                        else:
                            if (
                                allowed_gap < (start_offset - previous_end_offset)
                                or max_duration
                                < (previous_end_offset - current_start_offset)
                                or current_path != path
                            ):
                                if not re.search("×", transcript):
                                    writer.writerow(
                                        [
                                            current_path,
                                            transcript,
                                            previous_end_offset - current_start_offset,
                                            current_start_offset,
                                        ]
                                    )

                                current_start_offset = start_offset
                                current_path = path
                                transcript = word
                            else:
                                transcript += word

                        previous_segment_id = segment_id
                        previous_path = path
                        previous_end_offset = end_offset

                    if len(transcript) > 0:
                        if not re.search("×", transcript):
                            writer.writerow(
                                [
                                    current_path,
                                    transcript,
                                    end_offset - current_start_offset,
                                    current_start_offset,
                                ]
                            )


if __name__ == "__main__":
    create_dataset()
    print("✨ Done")
