import csv
import os
import re


def create_word(
    path_to_dataset="/workspace/datasets",
    out_dir="/workspace/datasets",
    eval_ids="A01M0110 A01M0137 A01M0097 A04M0123 A04M0121 A04M0051 A03M0156 A03M0112 "
    "A03M0106 A05M0011 A01M0056 A03F0072 A02M0012 A03M0016 A06M0064 A06F0135 "
    "A01F0034 A01F0063 A01F0001 A01M0141 S00M0112 S00F0066 S00M0213 S00F0019 "
    "S00M0079 S01F0105 S00F0152 S00M0070 S00M0008 S00F0148",
):
    """Create a word tsv file from .sdb files in MORPH directory

    Most codes are copied from csj_make_trans/csj2kaldi4m.pl in espnet(and espnet
    refers to kaldi). Original script was written in perl (meh).

    Note: Show output in japanese in terminal in Linux `iconv -f SJIS <file_name>`

    TODO: some other information like pronunciation may be necessary for training
    non-deep learning methods

    TODO: Although this script runs in about < 1min on i10-10900X, multiprocessing
    support may be necessary for users with non-workstation CPUs

    TODO: capitalize alphabets

    """
    train_out_file = "csj_train_word.tsv"
    val_out_file = "csj_val_word.tsv"
    eval_ids = eval_ids.split()

    with open(os.path.join(out_dir, train_out_file), "w") as train_out:
        with open(os.path.join(out_dir, val_out_file), "w") as val_out:
            train_writer = csv.writer(train_out, delimiter="\t")
            val_writer = csv.writer(val_out, delimiter="\t")

            for data_type in ["core", "noncore"]:
                directory = os.path.join(path_to_dataset, "MORPH", "SDB", data_type)

                if data_type == "core":
                    assert (
                        len(os.listdir(directory)) == 201
                    ), "Some files are missing in `core`"
                elif data_type == "noncore":
                    assert (
                        len(os.listdir(directory)) == 3101
                    ), "Some files are missing in `noncore`"

                for file in os.listdir(directory):
                    base_file = file.split(".")[0]
                    writer = val_writer if base_file in eval_ids else train_writer

                    with open(
                        os.path.join(directory, file), mode="r", encoding="shift_jis"
                    ) as f:
                        audio_path = os.path.join(
                            path_to_dataset,
                            "WAV",
                            data_type,
                            f"{base_file}.wav",
                        )

                        i = 0
                        p = 0

                        # note that time does not increment every step. (however, it
                        # does increase monotonically). some words are grouped and have
                        # the same time offset
                        for line in f:
                            is_skipword = False

                            line = line.rstrip()
                            line = re.split("\t", line)

                            info = line[3].split()  # `id <start>-<end> L:_-_`
                            segment_id = info[0]
                            time = info[1].split("-")
                            start_offset = float(time[0])
                            end_offset = float(time[1])

                            word = line[5]

                            # remove <ベル>, <FV>、<雑音>...
                            if re.search("^<.*>$", word):
                                continue

                            try:
                                num = line[9]  # number in japanese
                            except IndexError:
                                num = ""

                            try:
                                pos = line[11]  # 活用系
                            except IndexError:
                                pos = ""

                            # using right side of A -> (A x; o) except for some numbers
                            if re.search("A", word) or i != 0:
                                if (
                                    re.search("ゼロ|０|零|一|二|三|四|五|六|七|八|九|十|百|千|．", word)
                                    and re.search("A", word)
                                ) or (p == 1):
                                    p = 1
                                    # (A _; _) or _(A _; _)
                                    if re.search(";", word) and i == 0:
                                        word = num
                                    # _;_)
                                    elif re.search(";", word) and i != 0:
                                        word = num
                                        i = 0
                                        p = 0
                                    # (A _ or _ or .
                                    #  _ is a word in the left of ;
                                    else:
                                        word = num
                                        i += 1
                                else:
                                    # (A _; _)
                                    if re.search(";", word) and i == 0:
                                        word_type = re.split(";", word)
                                        word = word_type[1]
                                        word = re.sub("[\x00-\x7F]", "", word)
                                    # _;_)
                                    elif re.search(";", word) and i != 0:
                                        word_type = re.split(";", word)
                                        word = word_type[1]
                                        word = re.sub("[\x00-\x7F]", "", word)
                                        i = 0
                                    # (A _ or _
                                    #  _ is a word in the left of ;
                                    else:
                                        # not using the left side of ;
                                        is_skipword = True
                                        i += 1
                            else:
                                word = re.sub("[\x00-\x7F]", "", word)

                            word = re.sub("・$", "", word)
                            word = re.sub("んー+", "ん", word)
                            word = re.sub("ンー+", "ン", word)

                            # (? _) does not have `pos`
                            if not is_skipword and len(word) > 0 and pos:
                                writer.writerow(
                                    [
                                        audio_path,
                                        segment_id,
                                        word,
                                        start_offset,
                                        end_offset,
                                    ]
                                )


if __name__ == "__main__":
    create_word()
    print("✨ Done")
