import queue
from multiprocessing import Manager, Process

import soundcard as sc
from hydra.experimental import compose, initialize

import tensorflow as tf

# pulseaudio (Linux) does not work on forked proceeses
# ref: https://github.com/bastibe/SoundCard/issues/96

# TODO: right now, rnnt demo is only supported

multiprocessing = True

if __name__ == "__main__":
    initialize(config_path="../configs/rnnt", job_name="rnnt")
    cfgs = compose(config_name="librispeech_char.yml")

    sample_rate = cfgs.audio_feature.sample_rate  # 16000
    frame_len = cfgs.tflite.frame_length  # in seconds
    model_path = cfgs.tflite.model_path_to

    num_encoder_layers = cfgs.encoder.num_layers
    encoder_dim_model = cfgs.encoder.num_units

    num_predictor_layers = cfgs.predictor.num_layers
    predictor_dim_model = cfgs.predictor.dim_model

    chunk_size = int(frame_len * sample_rate)

    mics = sc.all_microphones(include_loopback=True)
    mic = sc.get_microphone(mics[0].id, include_loopback=True)

    if multiprocessing:
        m = Manager()
        q = m.Queue()

        def recognizer(q):
            model = tf.lite.Interpreter(model_path=model_path)

            input_details = model.get_input_details()
            output_details = model.get_output_details()

            model.resize_tensor_input(input_details[0]["index"], [chunk_size])
            model.allocate_tensors()

            def recognize(signal, prev_token, encoder_states, predictor_states):
                if signal.shape[0] < chunk_size:
                    signal = tf.pad(signal, [[0, chunk_size - signal.shape[0]]])

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
            encoder_states = tf.zeros(
                shape=[num_encoder_layers, 2, 1, encoder_dim_model],  # N_e, 2, B=1, D
                dtype=tf.float32,
            )
            predictor_states = tf.zeros(
                shape=[
                    num_predictor_layers,
                    2,
                    1,
                    predictor_dim_model,
                ],  # N_p, 2, B=1, D
                dtype=tf.float32,
            )
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

        def stream(q):
            global mic

            with mic.recorder(samplerate=sample_rate) as mic:
                while True:
                    audio_signals = mic.record(numframes=chunk_size)  # [frame, channel]
                    q.put(audio_signals[:, 0], timeout=frame_len)  # use first channel

        tflite_process = Process(target=recognizer, args=[q])
        tflite_process.start()

        send_process = Process(target=stream, args=[q])
        send_process.start()
        send_process.join()
        send_process.close()

        tflite_process.terminate()
    else:
        model = tf.lite.Interpreter(model_path=model_path)

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.resize_tensor_input(input_details[0]["index"], [chunk_size])
        model.allocate_tensors()

        prev_token = tf.zeros(shape=[], dtype=tf.int32)
        encoder_states = tf.zeros(
            shape=[num_encoder_layers, 2, 1, encoder_dim_model],  # N_e, 2, B=1, D
            dtype=tf.float32,
        )
        predictor_states = tf.zeros(
            shape=[num_predictor_layers, 2, 1, predictor_dim_model],  # N_p, 2, B=1, D
            dtype=tf.float32,
        )
        transcript = ""

        def recognize(signal, prev_token, encoder_states, predictor_states):
            if signal.shape[0] < chunk_size:
                signal = tf.pad(signal, [[0, chunk_size - signal.shape[0]]])

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

        with mic.recorder(samplerate=sample_rate) as mic:
            while True:
                data = mic.record(numframes=chunk_size)
                text, prev_token, encoder_states, predictor_states = recognize(
                    data[:, 0], prev_token, encoder_states, predictor_states
                )
                transcript += text
                print(transcript, flush=True)
