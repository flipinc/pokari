# PyTorch

## Getting Started
```shell
docker build -t transducer/pytorch -f docker/Dockerfile .
docker run --gpus all -it --rm -v ${PWD}:/workspace/pokari -v /home/keisuke26/Documents/Chief/Datasets/<path_to_dataset>:/workspace/datasets --shm-size=8g transducer/pytorch
```

## Production
Pytorch is the best for **developing** DL models. However, when it comes to deployment, it is much easier to use Tensorflow. Transducer models are quite complex because of its sequential and stateful structure, and I have not yet seen any successfully exported PyTorch models.
