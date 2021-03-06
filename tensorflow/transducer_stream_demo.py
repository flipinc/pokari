import queue
from multiprocessing import Manager, Process

import numpy as np
import soundcard as sc
from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer
from modules.emformer_encoder import EmformerEncoder
from modules.rnnt_encoder import RNNTEncoder

# pulseaudio (Linux) does not work on forked proceeses
# ref: https://github.com/bastibe/SoundCard/issues/96

multiprocessing = False

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="librispeech_char_mini_vgg.yml")

    # used for getting initial states
    transducer = Transducer(cfgs=cfgs, global_batch_size=1, setup_training=False)

    sample_rate = cfgs.audio_feature.sample_rate
    model_path = cfgs.tflite.model_path_to

    mics = sc.all_microphones(include_loopback=True)
    mic = sc.get_microphone(mics[0].id, include_loopback=True)

    if isinstance(transducer.encoder, EmformerEncoder):
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

        def record_emformer(callback):
            """Recording function for emformer

            Emformer uses future audio (right size) to commpute prediciton for the
            current audio (chunk size).

            """
            global mic

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

        record = record_emformer
    elif isinstance(transducer.encoder, RNNTEncoder):
        frame_len_in_second = cfgs.tflite.frame_length
        # segment size in number of frames
        segment_size = int(frame_len_in_second * sample_rate)

        def record_rnnt(callback):
            global mic

            with mic.recorder(samplerate=sample_rate) as mic:
                while True:
                    # [frame, channel]
                    audio_signals = mic.record(numframes=segment_size)
                    # use first channel
                    audio_signal = audio_signals[:, 0]

                    callback(audio_signal)

        record = record_rnnt
    else:
        NotImplementedError("Mock streaming is not implemented for this encoder.")

    if multiprocessing:
        m = Manager()
        q = m.Queue()

        def recognizer(q):
            model = tf.lite.Interpreter(model_path=model_path)

            input_details = model.get_input_details()
            output_details = model.get_output_details()

            model.resize_tensor_input(input_details[0]["index"], [segment_size])
            model.allocate_tensors()

            def recognize(signal, prev_token, encoder_states, predictor_states):
                if signal.shape[0] < segment_size:
                    signal = tf.pad(signal, [[0, segment_size - signal.shape[0]]])

                model.set_tensor(input_details[0]["index"], signal)
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

            prev_token = tf.zeros(shape=[], dtype=tf.int32)
            encoder_states = transducer.encoder.get_initial_state(batch_size=1)
            predictor_states = transducer.predictor.get_initial_state(batch_size=1)
            transcript = ""

            while True:
                try:
                    audio_signal = q.get()
                    text, prev_token, encoder_states, predictor_states = recognize(
                        audio_signal, prev_token, encoder_states, predictor_states
                    )
                    transcript += text
                    print(transcript, flush=True)
                except queue.Empty:
                    pass

        tflite_process = Process(target=recognizer, args=[q])
        tflite_process.start()

        send_process = Process(
            target=record,
            args=[
                lambda audio_signal: q.put(audio_signal, timeout=frame_len_in_second)
            ],
        )
        send_process.start()
        send_process.join()
        send_process.close()

        tflite_process.terminate()
    else:
        model = tf.lite.Interpreter(model_path=model_path)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.resize_tensor_input(input_details[0]["index"], [segment_size])
        model.allocate_tensors()

        prev_token = tf.zeros(shape=[], dtype=tf.int32)
        encoder_states = transducer.encoder.get_initial_state(batch_size=1)
        predictor_states = transducer.predictor.get_initial_state(batch_size=1)
        transcript = ""

        def recognize(signal, prev_token, encoder_states, predictor_states):
            if signal.shape[0] < segment_size:
                signal = tf.pad(signal, [[0, segment_size - signal.shape[0]]])

            model.set_tensor(input_details[0]["index"], signal)
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

        def callback(signal):
            global transcript, prev_token, encoder_states, predictor_states

            text, prev_token, encoder_states, predictor_states = recognize(
                signal, prev_token, encoder_states, predictor_states
            )
            transcript += text

            print(transcript, flush=True)

        record(callback)
