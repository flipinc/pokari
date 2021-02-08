# Pokari

![CI](https://github.com/chief-co-jp/pokari/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Start with Docker Compose
```shell
docker build -t ekaki/pokari .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/LibriSpeech:/workspace/datasets --shm-size=8g ekaki/pokari
```

./../Datasets/LibriSpeech/