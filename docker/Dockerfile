# Mainly from https://github.com/TensorSpeech/TensorFlowASR
FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3

RUN apt-get update && \
    apt-get -y install \
    # install sndfile library for librosa
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev \
    # install git
    git \
    # install cmake
    cmake protobuf-compiler

# install warprnnt. Using fork version of warp-transducer as the original
# version does not support tf 2.4 + CUDA 11 + compute_86 (RTX 3090)
# the author of discusses some problems here (https://github.com/TensorSpeech/TensorFlowASR/issues/54)
RUN git clone https://github.com/Thumb-Technologies/warp-transducer.git && \
    # set env flags
    export CUDA_HOME="/usr/local/cuda" && \
    # see https://github.com/baidu-research/warp-ctc/issues/119
    export TF_CXX11_ABI=1 && \
    # build warprnnt
    cd warp-transducer && \
    mkdir build && \
    cd build && \
    rm -f CMakeCache.txt && \
    cmake \
    -DUSE_NAIVE_KERNEL=on \
    -DCMAKE_C_COMPILER_LAUNCHER=$(which gcc) \
    -DCMAKE_CXX_COMPILER_LAUNCHER=$(which g++) \
    -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME .. && \
    make && \
    # install tensorflow binding
    cd ../tensorflow_binding && \
    CUDA=$CUDA_HOME python3 setup.py install && \
    rm -rf ../tests test ../pytorch_binding

# install pokari dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
# install tflite_runtime
# ref: https://www.tensorflow.org/lite/guide/python
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

WORKDIR /workspace/pokari
COPY . .