# Pokari

![CI](https://github.com/chief-co-jp/pokari/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Start with Docker
```shell
docker build -t ekaki/pokari .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/LibriSpeech:/workspace/datasets --shm-size=8g ekaki/pokari
```

### Design
- Since this repository is made for **production** use, every torch module should be scriptable meaning, wrapping by `torch.jit.script` should return `ScriptModule` without any error.
- Every `ScriptModule` should be instantiated at the root module (in most cases this is `LightningModule`), so that the converting the whole model to TorchScript is done by setting `return_torchscript` to `True`.