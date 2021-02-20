from hydra.experimental import compose, initialize

import tensorflow as tf
from models.transducer import Transducer

tf.keras.backend.clear_session()

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="librispeech_wordpiece.yml")

    tf.config.optimizer.set_experimental_options(
        {"auto_mixed_precision": cfgs.trainer.mxp}
    )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        visible_gpus = [gpus[i] for i in cfgs.trainer.devices]
        tf.config.set_visible_devices(visible_gpus, "GPU")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        global_batch_size = cfgs.trainer.batch_size
        global_batch_size *= strategy.num_replicas_in_sync

        transducer = Transducer(cfgs=cfgs, global_batch_size=global_batch_size)
        transducer._build()

        if "model_path" in cfgs.trainer:
            transducer.load_weights(cfgs.trainer.model_path)

        transducer._compile()
        transducer._fit()
