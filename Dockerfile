FROM python:3.7

WORKDIR /app

COPY requirements.txt .

# install pytorch
RUN pip install numpy
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# install other libraries
RUN pip install -r requirements.txt

# install sndfile library
RUN apt-get update \
    && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev -y \
    && pip install pyaudio

COPY . .