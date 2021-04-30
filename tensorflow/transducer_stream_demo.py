import soundcard as sc
from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer
from modules.emformer_encoder import EmformerEncoder
from modules.rnnt_encoder import RNNTEncoder

BATCH_SIZE = 2
INCLUDE_LOOPBACK = True

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="csj_char3265_mini_stack.yml")

    # text featurizer will not be used. to avoid import error, override this value
    cfgs.text_feature.vocab_path = None

    # used for getting initial states
    transducer = Transducer(
        cfgs=cfgs, global_batch_size=BATCH_SIZE, setup_training=False
    )

    sample_rate = cfgs.audio_feature.sample_rate

    # you can change this to use either mic or loopback
    mics = sc.all_microphones(include_loopback=INCLUDE_LOOPBACK)
    mic = sc.get_microphone(mics[0].id, include_loopback=INCLUDE_LOOPBACK)

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

            future_audio_signal = tf.zeros([right_size], dtype="float32")

            with mic.recorder(samplerate=sample_rate) as mic:
                while True:
                    # [frame, channel]
                    audio_signals = mic.record(numframes=chunk_size)
                    # use first channel
                    audio_signal = audio_signals[:, 0]
                    # concatenate with the last recorded audio
                    audio_signal = tf.concat(
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
                    # use the first channel
                    audio_signal = audio_signals[:, 0]

                    callback(audio_signal)

        record = record_rnnt
    else:
        NotImplementedError("Mock streaming is not implemented for this encoder.")

    model = tf.keras.models.load_model(cfgs.serve.model_path_to)

    if BATCH_SIZE > 1:
        prev_token = tf.zeros([BATCH_SIZE, 1], dtype=tf.int32)
        input_transcript = ""
        output_transcript = ""
    else:
        prev_token = tf.zeros([], dtype=tf.int32)
        transcript = ""

    encoder_states = transducer.encoder.get_initial_state(batch_size=BATCH_SIZE).numpy()
    predictor_states = transducer.predictor.get_initial_state(
        batch_size=BATCH_SIZE
    ).numpy()

    def recognize(signal, prev_token, encoder_states, predictor_states):
        if signal.shape[0] < segment_size:
            signal = tf.pad(signal, [[0, segment_size - signal.shape[0]]])

        # A batch version just simply replicates a loopback signal
        if BATCH_SIZE > 1:
            signal = tf.tile(tf.expand_dims(signal, axis=0), [BATCH_SIZE, 1])

        if BATCH_SIZE > 1:
            (
                upoints,
                prev_token,
                encoder_states,
                predictor_states,
            ) = model.stream_batch(signal, prev_token, encoder_states, predictor_states)
        else:
            (
                upoints,
                prev_token,
                encoder_states,
                predictor_states,
            ) = model.stream_one(signal, prev_token, encoder_states, predictor_states)

        if BATCH_SIZE > 1:
            text = [None] * BATCH_SIZE
            for i in range(BATCH_SIZE):
                text[i] = "".join([chr(u) for u in upoints[i]])
        else:
            text = "".join([chr(u) for u in upoints])

        return text, prev_token, encoder_states, predictor_states

    def callback(signal):
        global prev_token, encoder_states, predictor_states

        text, prev_token, encoder_states, predictor_states = recognize(
            signal, prev_token, encoder_states, predictor_states
        )

        if BATCH_SIZE > 1:
            global input_transcript, output_transcript

            input_transcript += text[0]
            output_transcript += text[1]
            print(input_transcript, flush=True)
        else:
            global transcript

            transcript += text
            print(transcript, flush=True)

    record(callback)
