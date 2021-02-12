# Pokari

![CI](https://github.com/chief-co-jp/pokari/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Start with Docker
```shell
docker build -t pokari/tensorflow -f docker/Dockerfile.tensorflow .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/LibriSpeech:/workspace/datasets --shm-size=8g pokari/tensorflow
```

### Design
- Pytorch is very easy to quickly develop a DL model. However, when it comes to deployment, especially around onnx support, it is much easier to use Tensorflow. Once Pytorch's support for ScriptModule -> onnx conversion is decent enough, I am going to think this over again. 