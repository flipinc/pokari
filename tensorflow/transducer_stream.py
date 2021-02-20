import pyaudio as pa
import soundcard as sc

p = pa.PyAudio()

# sample rate, Hz
SAMPLE_RATE = 16000

# duration of signal frame, seconds
FRAME_LEN = 1.6  # 1280ms + 320ms

CHANNELS = 1

CHUNK_SIZE = int(FRAME_LEN * SAMPLE_RATE)

mics = sc.all_microphones(include_loopback=True)
mic = sc.get_microphone(mics[0].id, include_loopback=True)

with mic.recorder(samplerate=16000) as mic:
    while True:
        data = mic.record(numframes=CHUNK_SIZE)
        print(data)
