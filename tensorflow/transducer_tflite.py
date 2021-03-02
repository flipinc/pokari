from hydra.experimental import compose, initialize
from omegaconf import DictConfig

import tensorflow as tf
from models.transducer import Transducer

tf.keras.backend.clear_session()


def convert_to_tflite(cfgs: DictConfig):
    transducer = Transducer(cfgs=cfgs, global_batch_size=1, setup_training=False)
    transducer._build()

    transducer.load_weights(cfgs.tflite.model_path_from)

    tf_func = transducer.make_one_tflite_function()
    concrete_func = tf_func.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    return tflite_model


if __name__ == "__main__":
    initialize(config_path="../configs/rnnt", job_name="rnnt")
    cfgs = compose(config_name="librispeech_char.yml")

    tflite_model = convert_to_tflite(cfgs)

    with open(cfgs.tflite.model_path_to, "wb") as tflite_out:
        tflite_out.write(tflite_model)

    print("🎉 Successfully converted to tflite model")
