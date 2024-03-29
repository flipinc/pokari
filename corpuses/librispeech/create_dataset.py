import csv
import json
import os


def create_tsv(
    out_file,
    path_to_dataset,
    out_dir="/workspace/datasets",
):
    """Create a tsv file which adheres to tensorflow format

    PATH\tTRANSCRIPT\tDURATION\tOFFSET

    """
    with open(os.path.join(out_dir, out_file), "w") as out:
        tsv_writer = csv.writer(out, delimiter="\t")

        for speaker in os.listdir(path_to_dataset):
            if not speaker.endswith(".DS_Store"):

                for topic in os.listdir(os.path.join(path_to_dataset, speaker)):
                    if not topic.endswith(".DS_Store"):

                        for content in os.listdir(
                            os.path.join(path_to_dataset, speaker, topic)
                        ):
                            if not content.endswith(".DS_Store"):
                                if content.endswith(".txt"):

                                    with open(
                                        os.path.join(
                                            path_to_dataset, speaker, topic, content
                                        ),
                                        "r",
                                    ) as txt_file:

                                        lines = [
                                            line.rstrip()
                                            for line in txt_file.readlines()
                                        ]

                                        for line in lines:
                                            id_and_text = line.split(None, 1)
                                            id = id_and_text[0]
                                            text = id_and_text[1]

                                            audio_path = os.path.join(
                                                path_to_dataset,
                                                speaker,
                                                topic,
                                                f"{id}.flac",
                                            )

                                            text = text.lower()

                                            tsv_writer.writerow(
                                                [audio_path, text, -1, 0.0]
                                            )


def create_manifest(
    out_file,
    path_to_dataset,
    out_dir="/workspace/datasets",
):
    """Create a manifest file which adheres to PyTorch NVIDIA/NeMo format

    - id
    - audio_file
    - duration # 0 by default
    - text
    - offset # 0 by default
    - speaker
    - orig_sr # None by default. This can be inferred by librosa.

    """
    with open(os.path.join(out_dir, out_file), "w") as out:

        for speaker in os.listdir(path_to_dataset):
            if not speaker.endswith(".DS_Store"):

                for topic in os.listdir(os.path.join(path_to_dataset, speaker)):
                    if not topic.endswith(".DS_Store"):

                        for content in os.listdir(
                            os.path.join(path_to_dataset, speaker, topic)
                        ):
                            if not content.endswith(".DS_Store"):
                                if content.endswith(".txt"):

                                    with open(
                                        os.path.join(
                                            path_to_dataset, speaker, topic, content
                                        ),
                                        "r",
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
                                                        path_to_dataset,
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


if __name__ == "__main__":
    # create_tsv(
    #     path_to_dataset="/workspace/datasets/train-large/train-clean-360",
    #     out_dir="/workspace/datasets",
    #     out_file="tensorflow_asr_360.tsv",
    # )

    # create_tsv(
    #     path_to_dataset="/workspace/datasets/dev/dev-clean",
    #     out_dir="/workspace/datasets",
    #     out_file="tensorflow_asr_val.tsv",
    # )

    # create_manifest(
    #     path_to_dataset="/workspace/datasets/train-large/train-clean-360",
    #     out_dir="/workspace/datasets",
    #     out_file="manifest_train_360.json",
    # )

    # create_manifest(
    #     path_to_dataset="/workspace/datasets/dev/dev-clean",
    #     out_dir="/workspace/datasets",
    #     out_file="manifest_val.json",
    # )
    pass
