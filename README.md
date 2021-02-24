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
Following installations are required after Docker run:
- tensorflow==2.3.2 (for tflite conversion only)
- tensorflow-io==0.16 (for tflite conversion only)

### TFLite Conversion
- As of 2021/2/17, tensorflow 2.4 does not work well with tflite. If you see errors such as 
`tensorflow.python.framework.errors_impl.InvalidArgumentError: Attempting to add a duplicate function with name`,
it is highly likely that reverting back tensorflow version might solve some issues (or at least give you directions to solve them). It is confirmed that tensorflow 2.3.2
works.
- tf.string is not supported in TFLite, so all models outputs Unicode instead.

### Limitations on Tensorflow Version
- Since warp_rnnt (optimized for CUDA for faster loss calculation) is not used, training time is much slower than PyTorch. It is desirable to use ![this library](https://github.com/HawkAaron/warp-transducer), but its required tensorflow version conflicts with RTX3090 which I am using for training.
- Gradient variational noise is not implemented yet

### Design
- Pytorch is very easy to develop a DL model. However, when it comes to deployment, especially around onnx support, it is much easier to use Tensorflow. Transducer model is quite complex because of its stateful structure, and I have not yet seen any successfully exported PyTorch models. For reference, CTC can be exported and ![NeMo](https://github.com/NVIDIA/NeMo/blob/25abffdd37efb3a9f5a6e236d910f045271ae08f/nemo/collections/asr/models/ctc_models.py) provides an interface for it.

### Run locally
- Only audio servers that support loopback recording are able to run locally (ie. Linux/Pulseaudio, Windows/WASAPI).
- To run TFLite models locally, install ![SoundCard](https://github.com/bastibe/SoundCard) by running `pip install soundcard`.