import os

import tensorflow as tf
from hydra.experimental import compose, initialize

from transducer import Transducer

tf.keras.backend.clear_session()

initialize(config_path="../configs/rnnt", job_name="rnnt")
cfgs = compose(config_name="librispeech_wordpiece.yml")

transducer = Transducer(cfgs=cfgs, global_batch_size=cfgs.tflite.batch_size)
tf_func = transducer.make_tflite_function()
concrete_func = tf_func.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

print("🎉 Successfully converted to tflite model")

tflite_path = "models/tflite"

if not os.path.exists(os.path.dirname(tflite_path)):
    os.makedirs(os.path.dirname(tflite_path))
with open(tflite_path, "wb") as tflite_out:
    tflite_out.write(tflite_model)
