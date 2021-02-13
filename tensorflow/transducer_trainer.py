from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer

if __name__ == "__main__":
    tf.executing_eagerly()

    initialize(config_path="../configs/emformer", job_name="emformer")
    cfg = compose(config_name="emformer_librispeech_char_tensorflow.yml")

    tf.keras.backend.clear_session()

    if "precision" in cfg:
        tf.mixed_precision.set_global_policy(
            tf.mixed_precision.Policy(**cfg["precision"])
        )

    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")s

    with strategy.scope():
        num_replicas = strategy.num_replicas_in_sync
        cfg.train_ds.batch_size *= num_replicas
        cfg.validation_ds.batch_size *= num_replicas

        model = Transducer(cfg=cfg)
        model.train(num_replicas=num_replicas)
