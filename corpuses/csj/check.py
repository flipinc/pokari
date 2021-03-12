def check_empty_transcript(paths_to_tsv):
    empty_ct = 0

    for path in paths_to_tsv:
        with open(path, "r") as f:
            for line in f:
                file, transcript, _, _ = line.split("\t")
                length = len(transcript)

                if length == 0:
                    print(f"🚨 `{file}` has empty transcript.")
                    empty_ct += 1

    if empty_ct > 0:
        print(f"🚨 Found {empty_ct} empty trascripts.")


def check_special_character(paths_to_tsv):
    pass


if __name__ == "__main__":
    paths_to_tsv = [
        "/workspace/datasets/csj_train.tsv",
        "/workspace/datasets/csj_val.tsv",
    ]
    check_empty_transcript(paths_to_tsv)
    print("✨ Done")
