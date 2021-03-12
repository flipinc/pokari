import librosa


def check_empty_transcript(paths_to_tsv):
    empty_ct = 0

    max_len = -1e5
    min_len = 1e5

    for path in paths_to_tsv:
        with open(path, "r") as f:
            for line in f:
                path, transcript, _, _ = line.split("\t")
                length = len(transcript)

                if max_len < length:
                    max_len = length

                if min_len > length:
                    min_len = length

                if length == 0:
                    print(f"🚨 `{path}` has empty transcript.")
                    empty_ct += 1

    print(f"📘 max transcript length is {max_len}.")
    print(f"📘 min transcript length is {min_len}.")

    if empty_ct > 0:
        print(f"🚨 Found {empty_ct} empty trascripts.")


def check_empty_audio(paths_to_tsv):
    empty_ct = 0

    max_len = -1e5
    min_len = 1e5

    for path in paths_to_tsv:
        with open(path, "r") as f:
            for line in f:
                path, _, duration, offset = line.split("\t")

                wave, rate = librosa.load(
                    path,
                    sr=16000,
                    mono=True,
                    duration=float(duration),
                    offset=float(offset),
                )
                length = wave.shape[0]

                if max_len < length:
                    max_len = length

                if min_len > length:
                    min_len = length

                if length == 0:
                    print(f"🚨 `{path}` has empty audio.")
                    empty_ct += 1

    print(f"📘 max audio length is {max_len}.")
    print(f"📘 min audio length is {min_len}.")

    if empty_ct > 0:
        print(f"🚨 Found {empty_ct} empty audio.")


if __name__ == "__main__":
    paths_to_tsv = [
        "/workspace/datasets/csj_train.tsv",
        "/workspace/datasets/csj_val.tsv",
    ]
    check_empty_transcript(paths_to_tsv)
    check_empty_audio(paths_to_tsv)
    print("✨ Done")
