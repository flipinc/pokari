import csv
import os
import re


def create_tsv(
    path_to_dataset="/workspace/datasets",
    out_dir="/workspace/datasets",
    eval_ids="A01M0110 A01M0137 A01M0097 A04M0123 A04M0121 A04M0051 A03M0156 A03M0112 "
    "A03M0106 A05M0011 A01M0056 A03F0072 A02M0012 A03M0016 A06M0064 A06F0135 "
    "A01F0034 A01F0063 A01F0001 A01M0141 S00M0112 S00F0066 S00M0213 S00F0019 "
    "S00M0079 S01F0105 S00F0152 S00M0070 S00M0008 S00F0148",
    max_duration=30,
):
    """Create a transcript tsv file of .sdb files in MORPH directory

    Number of characters/subwords to be used
    - CHR: 3084 -> https://arxiv.org/pdf/2102.07935.pdf
    - CHR: 3260 -> https://arxiv.org/pdf/2006.14941.pdf
    - CHR: 3262, 3500 -> http://www.interspeech2020.org/uploadfile/pdf/Mon-2-2-2.pdf
    - BPE: 7520 -> https://arxiv.org/pdf/2008.03822.pdf

    Show output in japanese in terminal in Linux `iconv -f SJIS <file_name>`

    PATH\tTRANSCRIPT\tDURATION\tOFFSET

    TODO: Most codes are copied from csj_make_trans/csj2kaldi4m.pl in espnet(and espnet
    refers to kaldi). However, since the original script was written in perl, some
    changes were made. These may cause bugs.

    TODO: some other information like pronunciation may be necessary for training
    non-deep learning methods

    TODO: Although this script runs in about < 1min on i10-10900X, multiprocessing
    support may be necessary

    """
    train_out_file = "csj_train.tsv"
    val_out_file = "csj_val.tsv"
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
                        current_words = []
                        current_offset = None
                        prev_end_offset = None

                        def save(end_offset):
                            writer.writerow(
                                [
                                    audio_path,
                                    "".join(current_words),
                                    end_offset - current_offset,
                                    current_offset,
                                ]
                            )

                        # note that time does not increment monotonically. some words
                        # are grouped and have the same time offset
                        for line in f:
                            line = line.rstrip()
                            line = re.split("\t", line)

                            time = (
                                line[3].split()[1].split("-")
                            )  # `id <start>-<end> L:_-_`
                            start_offset = float(time[0])
                            end_offset = float(time[1])
                            if current_offset is None:
                                current_offset = start_offset

                            if (end_offset - current_offset) > max_duration:
                                # does not include current word, so use previous end
                                # offset
                                save(prev_end_offset)
                                current_words = []
                                current_offset = start_offset

                            word = line[5]

                            if re.search("A", word) or i != 0:
                                if (
                                    re.search("ゼロ|０|零|一|二|三|四|五|六|七|八|九|十|百|千|．", word)
                                    and re.search("A", word)
                                ) or (p == 1):
                                    p = 1
                                    if re.search(";", word) and i == 0:
                                        word = line[9]
                                    elif re.search(";", word) and i != 0:
                                        word = line[9]
                                        i = 0
                                        p = 0
                                    else:
                                        word = line[9]
                                        i += 1
                                else:
                                    if re.search(";", word) and i == 0:
                                        word_type = re.split(";", word)
                                        # in original perl code, this is actually
                                        # `word_type[1]` but it prints overlapped
                                        # characters
                                        word = word_type[0]
                                        word = re.sub("[\x00-\x7F]", "", word)
                                    elif re.search(";", word) and i != 0:
                                        word_type = re.split(";", word)
                                        word = word_type[0]  # same as above
                                        word = re.sub("[\x00-\x7F]", "", word)
                                        i = 0
                                    else:
                                        word = re.sub("[\x00-\x7F]", "", word)
                                        i += 1
                            else:
                                word = re.sub("[\x00-\x7F]", "", word)

                            word = re.sub("・$", "", word)
                            word = re.sub(
                                "×", "", word
                            )  # this is not english character x
                            word = re.sub("雑音", "", word)
                            word = re.sub("笑", "", word)
                            word = re.sub("息", "", word)
                            word = re.sub("咳", "", word)
                            word = re.sub("泣", "", word)
                            word = re.sub("んー+", "ん", word)
                            word = re.sub("ンー+", "ン", word)

                            current_words.append(word)
                            prev_end_offset = end_offset

                        # if there are some words left, save
                        if len(current_words) > 0:
                            save(end_offset)


def create_vocab(
    paths_to_tsv=[
        "/workspace/datasets/csj_train.tsv",
        "/workspace/datasets/csj_val.tsv",
    ],
    out_dir="/workspace/datasets",
    num_vocabs=3224,
):
    """Create a Japanese vocabulary from given tsv file

    CSJ has 3224 characters in total

    """
    vocab_freq = {}
    out_file = f"jp_{num_vocabs}.char"

    with open(os.path.join(out_dir, out_file), "w") as out:
        for path in paths_to_tsv:
            with open(path, "r") as f:
                for line in f:
                    line = line.rstrip()
                    transcript = line.split("\t")[1]
                    for word in transcript:
                        if word in vocab_freq:
                            vocab_freq[word] += 1
                        else:
                            vocab_freq[word] = 1

        vocabs = [
            k
            for k, _ in sorted(
                vocab_freq.items(), key=lambda item: item[1], reverse=True
            )
        ]
        vocabs = sorted(vocabs[:num_vocabs])

        for char in vocabs:
            out.write("%s\n" % char)


if __name__ == "__main__":
    # create_tsv()
    # create_vocab()
    print("✨ Done")
