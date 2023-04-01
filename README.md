# Pokari | Streaming ASRs

![CI](https://github.com/chief-co-jp/pokari/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Tensorflow

### Getting Started

```shell
docker build -t transducer/tensorflow -f docker/Dockerfile.tensorflow .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v <path_to_dataset>:/workspace/datasets --shm-size=1g --ulimit memlock=-1 transducer/tensorflow
```

To open tensorboard, run

```shell
tensorboard --logdir=outputs/tensorflow/logs
```

### Saving as a `SavedModel`

As of 2021/4/30, tensorflow 2.4 gives an ![error](https://github.com/tensorflow/tensorflow/issues/44541). The SavedModel just runs fine but if you want to eliminate the error, following (un)installations are necessary:

```shell
pip uninstall warprnnt_tensorflow
pip install tensorflow==2.3.2 tensorflow-io==0.16 tensorflow-text==2.3
```

You can inspect the SavedModel with a following command inside Docker

```shell
saved_model_cli show --dir ./outputs/tensorflow/savedmodels/<MODEL_NAME> --all
```

### Upload

1. Configure AWS keys
   Run `aws configure` and follow its messages.
2. Modify and run `tensorflow/save_to_s3.py`

### Limitations

- CTC training is much slower compared to Transducer trianing which uses warprnnt-tensorflow for loss computation
- Mixed precision training is not supported yet
- Can only save keras.Model with `save_traces=True` (all `get_config`s are not used)

## TFLite

**TFLite support has ended since v0.1.0. All related codes will be removed gradually.**

### Why we abandoned TFLite

- SavedModel and TFLite are not that diffrent. Both required great amount of engineering but TFLite requires more (e.g. unsupported operations)
- Scaling instances that's running TFLite models is extremely hard whereas Tensorflow Serving implements batch processing
- TFLite doesn't run on GPU

### Getting Started

Following (un)installations are required for tflite conversion:

```shell
pip uninstall warprnnt_tensorflow
pip install tensorflow==2.3.2 tensorflow-io==0.16 tensorflow-text==2.3
```

### TFLite Conversion

- As of 2021/2/17, tensorflow 2.4 does not work well with tflite. If you see errors such as
  `tensorflow.python.framework.errors_impl.InvalidArgumentError: Attempting to add a duplicate function with name`,
  it is highly likely that reverting back tensorflow version might solve some issues (or at least give you directions to solve them). It is confirmed that following tensorflow version works. - tensorflow 2.3.2
- tf.string is not supported in TFLite, so all models outputs Unicode instead.
- Make sure the converter version matches with the runtime version to align operation versions

### Running Demo

Since a live demo uses OS's audio APIs, first, make sure you are NOT running on a docker container. Only audio APIs that support loopback recording are able to run locally (ie. Linux/Pulseaudio, Windows/WASAPI).

```shell
pip install SoundCard
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
python3 tensorflow/transducer_stream_demo.py
```

## PyTorch

### Getting Started

```shell
docker build -t transducer/pytorch -f docker/Dockerfile .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v <path_to_dataset>:/workspace/datasets --shm-size=8g transducer/pytorch
```

### Production

Pytorch is the best for **developing** DL models. However, when it comes to deployment, it is much easier to use Tensorflow. Transducer models are quite complex because of its sequential and stateful structure, and I have not yet seen any successfully exported PyTorch models.

## Supported Datasets

- LibriSpeech (English)
- CSJ (Japanese)
