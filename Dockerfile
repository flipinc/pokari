FROM python:3.8

WORKDIR /app

COPY requirements.txt .

# install pytorch
RUN pip install numpy
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# install other libraries
RUN pip install -r requirements.txt

COPY . .