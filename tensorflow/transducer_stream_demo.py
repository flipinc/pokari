import queue
from multiprocessing import Manager, Process

import pyaudio as pa
import soundcard as sc
from hydra.experimental import compose, initialize

import tensorflow as tf

initialize(config_path="../configs/emformer", job_name="emformer")
cfgs = compose(config_name="librispeech_char_mini_vgg.yml")

if __name__ == "__main__":
    SAMPLE_RATE = cfgs.tflite.sample_rate  # 16000
    FRAME_LEN = cfgs.tflite.frame_length  # 1280ms + 320ms = 1.6
    CHANNELS = cfgs.tflite.channels  # 1
    MODEL_PATH = cfgs.tflite.model_path_to

    NUM_PREDICTOR_LAYERS = cfgs.predictor.num_layers
    PREDICTOR_DIM_MODEL = cfgs.predictor.dim_model

    CHUNK_SIZE = int(FRAME_LEN * SAMPLE_RATE)

    p = pa.PyAudio()

    mics = sc.all_microphones(include_loopback=True)
    mic = sc.get_microphone(mics[0].id, include_loopback=True)

    m = Manager()
    mq = m.Queue()

    def recognizer(mq):
        """Run inference on tflite model"""
        model = tf.lite.Interpreter(model_path=MODEL_PATH)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.resize_tensor_input(input_details[0]["index"], [CHUNK_SIZE])
        model.allocate_tensors()

        def recognize(signal, prev_token, states):
            if signal.shape[0] < CHUNK_SIZE:
                signal = tf.pad(signal, [[0, CHUNK_SIZE - signal.shape[0]]])

            model.set_tensor(input_details[0]["index"], signal)
            model.set_tensor(input_details[1]["index"], prev_token)
            model.set_tensor(input_details[2]["index"], states)

            model.invoke()

            upoints = model.get_tensor(output_details[0]["index"])
            prev_token = model.get_tensor(output_details[1]["index"])
            states = model.get_tensor(output_details[2]["index"])

            text = "".join([chr(u) for u in upoints])

            return text, prev_token, states

        prev_token = tf.zeros(shape=[], dtype=tf.int32)
        states = tf.zeros(
            shape=[NUM_PREDICTOR_LAYERS, 2, 1, PREDICTOR_DIM_MODEL],  # N, 2, B, D
            dtype=tf.float32,
        )
        transcript = ""

        while True:
            try:
                data = mq.get()
                text, prev_token, states = recognize(data, prev_token, states)
                transcript += text
                print(transcript, flush=True)
            except queue.Empty:
                pass

    def streamer(mq):
        global mic

        with mic.recorder(samplerate=SAMPLE_RATE) as mic:
            while True:
                data = mic.record(numframes=CHUNK_SIZE)
                mq.put(data)

    tflite_process = Process(target=recognizer, args=[mq])
    tflite_process.start()

    audio_process = Process(target=streamer, args=[mq])
    audio_process.start()

    audio_process.join()
    audio_process.close()

    tflite_process.terminate()
