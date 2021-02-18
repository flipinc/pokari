# Pokari

![CI](https://github.com/chief-co-jp/pokari/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Start with Docker
```shell
docker build -t docker/tensorflow -f docker/Dockerfile .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/LibriSpeech:/workspace/datasets --shm-size=8g docker/tensorflow
```
Followings installation are required after Docker run:
- tensorflow-datasets==4.2.0
- tensorflow==2.3 (for tflite conversion only)
- tensorflow-io==0.16 (for tflite conversion only)

### TFLite Conversion
- As of 2021/2/17, tensorflow 2.4 does not work well with tflite. If you see errors such as 
`tensorflow.python.framework.errors_impl.InvalidArgumentError: Attempting to add a duplicate function with name`,
it is highly likely that reverting back tensorflow version might solve some issues (or at least give you directions to solve them). It is confirmed that tensorflow 2.3 
works.
- tf.string is not supported in TFLite, so all models outputs Unicode instead.

### Design
- Pytorch is very easy to quickly develop a DL model. However, when it comes to deployment, especially around onnx support, it is much easier to use Tensorflow. Once Pytorch's support for ScriptModule -> onnx conversion is decent enough, I am going to think this over again. 