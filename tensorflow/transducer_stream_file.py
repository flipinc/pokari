from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer

tf.keras.backend.clear_session()

if __name__ == "__main__":
    initialize(config_path="../configs/rnnt", job_name="rnnt")
    cfgs = compose(config_name="librispeech_char.yml")

    transducer = Transducer(cfgs=cfgs, global_batch_size=1)
    transducer._build()

    transducer.load_weights(cfgs.trainer.model_path)

    transducer.stream()
