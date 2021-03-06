# Pokari

![CI](https://github.com/chief-co-jp/pokari/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Start with PyTorch Docker
```shell
docker build -t transducer/pytorch -f docker/Dockerfile .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/LibriSpeech:/workspace/datasets --shm-size=8g transducer/pytorch
```

### Start with Tensorflow Docker
```shell
docker build -t transducer/tensorflow -f docker/Dockerfile .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/LibriSpeech:/workspace/datasets --shm-size=1g --ulimit memlock=-1 transducer/tensorflow
```
Following (un)installations are required after Docker run for tflite conversion:
```shell
pip uninstall warprnnt_tensorflow
pip install tensorflow==2.3.2 tensorflow-io==0.16
```

### Run Demo on Tensorflow
Two streaming options are provided: 1) Live demo and 2) Mock streaming using a sample audio file.
1) Live demo
Since live demo uses OS audio API, first, make sure you are not running on docker container. Only audio servers that support loopback recording are able to run locally (ie. Linux/Pulseaudio, Windows/WASAPI). Multiprocessing (audio capture & tflite inference) is only supported on Windows. For linux, change to `multiprocessing=False`.
```shell
pip install SoundCard
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
python3 tensorflow/transducer_stream_demo.py
```
2) Mock streaming using a sample audio file
As of now, only loading from one of training examples are supported. Will support loading arbitrary file(s) given as args in the future.
```shell
python3 tensorflow/transducer_stream_file.py
```

### TFLite Conversion
- As of 2021/2/17, tensorflow 2.4 does not work well with tflite. If you see errors such as 
`tensorflow.python.framework.errors_impl.InvalidArgumentError: Attempting to add a duplicate function with name`,
it is highly likely that reverting back tensorflow version might solve some issues (or at least give you directions to solve them). It is confirmed that following tensorflow version works.
    - tensorflow 2.3.2
- tf.string is not supported in TFLite, so all models outputs Unicode instead.
- Make sure the converter version matches with the runtime version to align operation versions

### Limitations on Tensorflow Version
- CTC training is much slower compared to Transducer trianing which uses warprnnt-tensorflow for loss computation
- Mixed precision training is not supported yet

### Advice
- To keep tabs on which commit generated which result, when a training is finished, post a screen-shot of training loss curve, its commit number, and a configuration file used for it.

### Design
- Pytorch is the best for developing a DL model. However, when it comes to deployment, it is much easier to use Tensorflow. Transducer models are quite complex because of its stateful structure, and I have not yet seen any successfully exported PyTorch models. Therefore, Tensorflow is used as a default developing framework.