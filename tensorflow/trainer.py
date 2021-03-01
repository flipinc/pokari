from hydra.experimental import compose, initialize
from hydra.utils import instantiate

import tensorflow as tf

tf.keras.backend.clear_session()

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="librispeech_subword_1024.yml")

    if "mxp" in cfgs.trainer:
        # TODO: support mixed precision training. All layers must adapt to
        # mixed precision including transducer loss.

        # policy = tf.keras.mixed_precision.Policy(cfgs.trainer.mxp)
        # tf.keras.mixed_precision.set_global_policy(policy)
        pass

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        visible_gpus = [gpus[i] for i in cfgs.trainer.devices]
        tf.config.set_visible_devices(visible_gpus, "GPU")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        global_batch_size = cfgs.trainer.batch_size
        global_batch_size *= strategy.num_replicas_in_sync

        model = instantiate(
            cfgs.base_model, cfgs=cfgs, global_batch_size=global_batch_size
        )
        model._build()

        if "model_path" in cfgs.trainer:
            print(f"Loading from {cfgs.trainer.model_path} ...")
            model.load_weights(cfgs.trainer.model_path)

        model._compile()
        model._fit()
