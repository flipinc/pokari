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

## Save in a SavedModel format
As of 2021/4/30, tensorflow 2.4 gives an ![error](https://github.com/tensorflow/tensorflow/issues/44541). The SavedModel just runs fine but if you want to eliminate the error, following (un)installations are necessary:
```shell
pip uninstall warprnnt_tensorflow
pip install tensorflow==2.3.2 tensorflow-io==0.16 tensorflow-text==2.3
```
You can inspect the SavedModel with a following command inside Docker
```shell
saved_model_cli show --dir ./outputs/tensorflow/savedmodels/<MODEL_NAME> --all
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
- Can only save keras.Model with `save_traces=True` (all `get_config`s are not used)