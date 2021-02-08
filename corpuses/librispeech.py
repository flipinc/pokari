import json
import os


def create_manifest(
    path_to_libri="drive/MyDrive/jynote/LibriSpeech",
    kind="train-clean-100",
    out_dir="drive/MyDrive/jynote",
    out_file="manifest_train.json",
):
    """
    Create a librispeech manifest file containing following data for each audio data:
    - id
    - audio_file
    - duration # 0 by default
    - text
    - offset # 0 by default
    - speaker
    - orig_sr # None by default. This can be inferred by librosa.

    """
    root = os.path.join(path_to_libri, kind)

    with open(os.path.join(out_dir, out_file), "w") as out:

        for speaker in os.listdir(root):
            if not speaker.endswith(".DS_Store"):

                for topic in os.listdir(os.path.join(root, speaker)):
                    if not topic.endswith(".DS_Store"):

                        for content in os.listdir(os.path.join(root, speaker, topic)):
                            if not content.endswith(".DS_Store"):
                                if content.endswith(".txt"):

                                    with open(
                                        os.path.join(root, speaker, topic, content), "r"
                                    ) as txt_file:
                                        lines = [
                                            line.rstrip()
                                            for line in txt_file.readlines()
                                        ]
                                        for line in lines:
                                            id_and_text = line.split(None, 1)
                                            id = id_and_text[0]
                                            text = id_and_text[1]

                                            json.dump(
                                                {
                                                    "id": id,
                                                    "audio_filepath": os.path.join(
                                                        root,
                                                        speaker,
                                                        topic,
                                                        f"{id}.flac",
                                                    ),
                                                    "text": text.lower(),
                                                    "speaker": speaker,
                                                    "duration": 0.0,
                                                },
                                                out,
                                            )
                                            out.write("\n")
