import os


def create_vocab(
    paths_to_tsv=[
        "/workspace/datasets/csj_train.tsv",
        "/workspace/datasets/csj_val.tsv",
    ],
    out_dir="/workspace/datasets",
    num_vocabs=3265,
):
    """Create a Japanese vocabulary from given tsv file

    CSJ has 3265 characters in total

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
    create_vocab()
    print("✨ Done")
