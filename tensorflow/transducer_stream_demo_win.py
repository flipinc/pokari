import queue
from multiprocessing import Manager, Process

import soundcard as sc

# This demo only works on Windows since Pulseaudio (Linux) does not work on forked
# proceeses.
# ref: https://github.com/bastibe/SoundCard/issues/96

# TODO: this is not complete yet. add tflite interpreter

if __name__ == "__main__":

    def recognize(q_input, q_output):
        while True:
            try:
                input = q_input.get()
            except queue.Empty:
                input = None

            try:
                output = q_output.get()
            except queue.Empty:
                output = None

        print("A: ", input)
        print("B: ", output)

    def record(mic, queue):
        with mic.recorder(samplerate=16000) as mic:
            while True:
                audio_signals = mic.record(numframes=32 * 4 * 160)
                audio_signal = audio_signals[:, 0]
                queue.put(audio_signal)

    mics = sc.all_microphones(include_loopback=True)
    input_mic = sc.get_microphone(mics[0].id, include_loopback=True)
    print(f"Using {mics[0]} as input")
    output_mic = sc.get_microphone(mics[1].id, include_loopback=True)
    print(f"Using {mics[1]} as output")

    m = Manager()
    q_input = m.Queue()
    q_output = m.Queue()

    recognize_process = Process(target=recognize, args=[q_input, q_output])
    recognize_process.start()

    input_process = Process(target=record, args=[input_mic, q_input])
    output_process = Process(target=record, args=[output_mic, q_output])

    input_process.start()
    output_process.start()

    input_process.join()
    output_process.join()

    input_process.close()
    output_process.close()

    recognize_process.terminate()
