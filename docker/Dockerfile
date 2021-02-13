# as of 20212/13, `latest` has some bugs related to RTX 3090 configuration
# however, changing to nightly will cause some bugs in warprnnt
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

# see https://github.com/psf/black/issues/1707
RUN pip uninstall -y typing

# install pokari dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace/pokari
COPY . .



