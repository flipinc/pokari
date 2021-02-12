FROM tensorflow/tensorflow:latest-gpu

# install sndfile library
RUN apt-get update && \
    apt-get -y install \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev

# install git
RUN apt-get update && \
    apt-get -y install \
    git

# install cmake
RUN apt-get update && \
    apt-get -y install \
    cmake protobuf-compiler

# install nano
RUN apt-get update && \
    apt-get -y install \
    nano

# install warprnnt
# ref: https://github.com/HawkAaron/warp-transducer/issues/15
RUN git clone https://github.com/HawkAaron/warp-transducer && \
    cd warp-transducer && \
    # build warprnnt
    mkdir build && \
    cd build && \
    rm -f CMakeCache.txt && \
    cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME .. && \
    make VERBOSE=1 && \
    # set env flags
    export CUDA_HOME="/usr/local/cuda" && \
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
    # install pytorch binding
    cd ../tensorflow_binding && \
    python3 setup.py install && \
    rm -rf ../tests test ../pytorch_binding

# see https://github.com/psf/black/issues/1707
RUN pip uninstall -y typing

# install pokari dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace/pokari
COPY . .

# Note: After docker built, you must cd into the following directory 
# /usr/local/lib/python3.6/dist-packages/warprnnt_tensorflow-0.1-py3.6-linux-x86_64.egg/warprnnt_tensorflow
# and edit `__init__.py` by commenting out everything after RegisterShape
# ref: https://github.com/TensorSpeech/TensorFlowASR/issues/88
# ref: https://github.com/HawkAaron/warp-transducer/issues/59
# ref: https://stackoverflow.com/questions/59043649/attributeerror-module-tensorflow-python-framework-ops-has-no-attribute-regis



