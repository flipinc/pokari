# as of 2021/2/13, tensorflow/tensorflow:nightly-gpu or :latest-gpu has ome bugs related to RTX 3090 configuration
# probably because it does not use CUDA 11.1
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

# install sndfile library for librosa
RUN apt-get update && \
    apt-get -y install \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev

WORKDIR /tmp/pokari
# install pokari dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# see https://github.com/tensorflow/tensorflow/issues/43947
RUN ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11 $(python -c "import tensorflow.python as x; print(x.__path__[0])")/libcusolver.so.10
RUN ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcupti.so.11.1 $(python -c "import tensorflow.python as x; print(x.__path__[0])")/libcupti.so.11.0

WORKDIR /workspace/pokari
COPY . .