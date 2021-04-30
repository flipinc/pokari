from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer

tf.keras.backend.clear_session()

TYPE = 1

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="csj_char3265_mini_stack.yml")

    if TYPE == 0:  # stream_file
        transducer = Transducer(cfgs=cfgs, global_batch_size=1, setup_training=True)
        transducer._build()

        transducer.load_weights(cfgs.trainer.model_path)

        transducer.stream_file()

    elif TYPE == 1:  # stream_batch
        transducer = Transducer(cfgs=cfgs, global_batch_size=2, setup_training=False)
        transducer._build()

        transducer.load_weights(cfgs.trainer.model_path)

        transducer.stream_batch(
            tf.zeros([2, 25600], tf.float32),
            tf.zeros([2, 1], tf.int32),
            tf.zeros([2, 18, 2, 20, 8, 64], tf.float32),
            tf.zeros([1, 2, 2, 512], tf.float32),
        )

    print("✨ Done.")
