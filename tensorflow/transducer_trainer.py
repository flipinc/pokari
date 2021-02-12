from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer

initialize(config_path="configs/emformer", job_name="emformer")
cfg = compose(config_name="emformer_librispeech_char.yaml")

tf.keras.backend.clear_session()

policy = tf.mixed_precision.Policy("mixed_float16")
tf.mixed_precision.set_global_policy(policy)

strategy = None

# TODO: fix https://github.com/tensorflow/tensorflow/issues/44777
# seems like LD_LIBRARY_PATH is the root cause

with strategy.scope():
    model = Transducer(cfg=cfg.model)
    model.summary()
    model.train(cfg=cfg.trainer)
