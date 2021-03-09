import queue
from multiprocessing import Manager, Process

import numpy as np
import soundcard as sc
import tflite_runtime.interpreter as tflite
from hydra.experimental import compose, initialize

from models.transducer import Transducer

# pulseaudio (Linux) does not work on forked proceeses
# ref: https://github.com/bastibe/SoundCard/issues/96

multiprocessing = False

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="librispeech_char_mini_vgg.yml")

    # used for getting initial states
    transducer = Transducer(cfgs=cfgs, global_batch_size=2, setup_training=False)

    sample_rate = cfgs.audio_feature.sample_rate
    model_path = cfgs.tflite.model_path_to

    mics = sc.all_microphones(include_loopback=False)
    input_mic = sc.get_microphone(mics[0].id, include_loopback=False)

    mics = sc.all_microphones(include_loopback=True)
    output_mic = sc.get_microphone(mics[1].id, include_loopback=True)

    # right size in number of frames
    right_size = (
        transducer.encoder.right_length
        * transducer.encoder.subsampling_factor
        * transducer.audio_featurizer.hop_length
    )
    # chunk size in number of frames
    chunk_size = (
        transducer.encoder.chunk_length
        * transducer.encoder.subsampling_factor
        * transducer.audio_featurizer.hop_length
    )

    segment_size = right_size + chunk_size

    frame_len_in_second = int(chunk_size // 16000)

    model = tflite.Interpreter(model_path=model_path)

    def recognizer(q_input, q_output):
        model = tflite.Interpreter(model_path=model_path)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.resize_tensor_input(input_details[0]["index"], [2, segment_size])
        model.allocate_tensors()

        def recognize(signals, prev_token, encoder_states, predictor_states):
            model.set_tensor(input_details[0]["index"], signals)
            model.set_tensor(input_details[1]["index"], prev_token)
            model.set_tensor(input_details[2]["index"], encoder_states)
            model.set_tensor(input_details[3]["index"], predictor_states)

            model.invoke()

            upoints = model.get_tensor(output_details[0]["index"])
            prev_token = model.get_tensor(output_details[1]["index"])
            encoder_states = model.get_tensor(output_details[2]["index"])
            predictor_states = model.get_tensor(output_details[3]["index"])

            text = "".join([chr(u) for u in upoints])

            return text, prev_token, encoder_states, predictor_states

        prev_token = np.zeros(shape=[2, 1], dtype=np.int32)
        encoder_states = transducer.encoder.get_initial_state(batch_size=2).numpy()
        predictor_states = transducer.predictor.get_initial_state(batch_size=2).numpy()
        input_transcript = ""
        output_transcript = ""

        while True:
            try:
                input_audio_signal = q_input.get()
                if input_audio_signal.shape[0] < segment_size:
                    input_audio_signal = np.pad(
                        input_audio_signal,
                        (0, segment_size - input_audio_signal.shape[0]),
                    )
            except queue.Empty:
                input_audio_signal = np.zeros(shape=[segment_size], dtype=np.float32)

            try:
                output_audio_signal = q_output.get()
                if output_audio_signal.shape[0] < segment_size:
                    output_audio_signal = np.pad(
                        output_audio_signal,
                        (0, segment_size - output_audio_signal.shape[0]),
                    )
            except queue.Empty:
                output_audio_signal = np.zeros(shape=[segment_size], dtype=np.float32)

            audio_signals = np.stack([input_audio_signal, output_audio_signal])
            text, prev_token, encoder_states, predictor_states = recognize(
                audio_signals, prev_token, encoder_states, predictor_states
            )
            input_transcript += text[0]
            output_transcript += text[1]
            print("💁‍♂️: ", input_transcript, flush=True)
            print("💁‍♀️: ", output_transcript, flush=True)

    def record_emformer(mic, callback):
        """Recording function for emformer

        Emformer uses future audio (right size) to commpute prediciton for the
        current audio (chunk size).

        """
        future_audio_signal = np.zeros(right_size, dtype="float32")

        with mic.recorder(samplerate=sample_rate) as mic:
            while True:
                # [frame, channel]
                audio_signals = mic.record(numframes=chunk_size)
                # use first channel
                audio_signal = audio_signals[:, 0]
                # concatenate with the last recorded audio
                audio_signal = np.concatenate(
                    [future_audio_signal, audio_signal], axis=0
                )
                # cache the end of recorded audio
                future_audio_signal = audio_signal[-right_size:]

                callback(audio_signal)

    m = Manager()
    q_input = m.Queue()
    q_output = m.Queue()

    tflite_process = Process(target=recognizer, args=[q_input, q_output])
    tflite_process.start()

    input_process = Process(
        target=record_emformer,
        args=[
            input_mic,
            lambda audio_signal: q_input.put(audio_signal, timeout=frame_len_in_second),
        ],
    )

    output_process = Process(
        target=record_emformer,
        args=[
            output_mic,
            lambda audio_signal: q_output.put(
                audio_signal, timeout=frame_len_in_second
            ),
        ],
    )

    input_process.join()
    output_process.join()

    input_process.close()
    output_process.close()

    tflite_process.terminate()
