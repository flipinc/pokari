# import queue
# from multiprocessing import Manager, Process

# import numpy as np
import soundcard as sc

if __name__ == "__main__":
    mics = sc.all_microphones(include_loopback=False)
    input_mic = sc.get_microphone(mics[0].id, include_loopback=False)

    print("Input: ", mics)

    mics = sc.all_microphones(include_loopback=True)
    output_mic = sc.get_microphone(mics[1].id, include_loopback=True)

    print("Output: ", mics)

    # sample_rate = 16000
    # frame_len_in_second = 1.6
    # segment_size = int(frame_len_in_second * sample_rate)

    # def responder(q_input, q_output):
    #     while True:
    #         try:
    #             input_audio_signal = q_input.get()
    #             if input_audio_signal.shape[0] < segment_size:
    #                 input_audio_signal = np.pad(
    #                     input_audio_signal,
    #                     (0, segment_size - input_audio_signal.shape[0]),
    #                 )
    #         except queue.Empty:
    #             input_audio_signal = np.zeros(shape=[segment_size], dtype=np.float32)

    #         try:
    #             output_audio_signal = q_output.get()
    #             if output_audio_signal.shape[0] < segment_size:
    #                 output_audio_signal = np.pad(
    #                     output_audio_signal,
    #                     (0, segment_size - output_audio_signal.shape[0]),
    #                 )
    #         except queue.Empty:
    #             output_audio_signal = np.zeros(shape=[segment_size], dtype=np.float32)

    #         audio_signals = np.stack([input_audio_signal, output_audio_signal])

    #         print(audio_signals.shape)

    # def record(mic, callback):
    #     with mic.recorder(samplerate=sample_rate) as mic:
    #         while True:
    #             # [frame, channel]
    #             audio_signals = mic.record(numframes=segment_size)
    #             # use first channel
    #             audio_signal = audio_signals[:, 0]

    #             callback(audio_signal)

    # m = Manager()
    # q_input = m.Queue()
    # q_output = m.Queue()

    # res_process = Process(target=responder, args=[q_input, q_output])
    # res_process.start()

    # input_process = Process(
    #     target=record,
    #     args=[
    #         input_mic,
    #         lambda audio_signal: q_input.put(
    #             audio_signal, timeout=frame_len_in_second
    #         ),
    #     ],
    # )
    # output_process = Process(
    #     target=record,
    #     args=[
    #         output_mic,
    #         lambda audio_signal: q_output.put(
    #             audio_signal, timeout=frame_len_in_second
    #         ),
    #     ],
    # )

    # input_process.join()
    # output_process.join()

    # input_process.close()
    # output_process.close()

    # res_process.terminate()
