# ref:https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html

import boto3

if __name__ == "__main__":
    file_path = "/workspace/pokari/tflites/rnnt_char.tflite"
    bucket = "ekaki-pokari"

    object_name = file_path.split("/")[-1]

    s3_client = boto3.client("s3")
    response = s3_client.upload_file(file_path, bucket, object_name)

    print("✨ Done")
