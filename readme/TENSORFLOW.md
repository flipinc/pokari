# Tensorflow

## Getting Started
```shell
docker build -t transducer/tensorflow -f docker/Dockerfile.tensorflow .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v <absolute_path_to_dataset>:/workspace/datasets --shm-size=1g --ulimit memlock=-1 transducer/tensorflow
```
To open tensorboard, run
```shell
tensorboard --logdir=outputs/tensorflow/logs
```

## Demo
As of now, only loading a sample audio file from one of training examples are supported. We will support loading arbitrary file(s) in the future.
```shell
python3 tensorflow/scripts/transducer_stream_file.py
```

## Upload
1) Configure AWS keys
Run `aws configure` and follow its messages.
2) Modify and run `tensorflow/scripts/save_to_s3.py`

## Limitations
- CTC training is much slower compared to Transducer trianing which uses warprnnt-tensorflow for loss computation
- Mixed precision training is not supported yet