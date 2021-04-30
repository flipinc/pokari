from hydra.experimental import compose, initialize

from models.transducer import Transducer

# see followings for detailed information on savedmodels
# https://www.tensorflow.org/guide/saved_model
# https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model?hl=ja
# https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html
# https://github.com/keras-team/keras/issues/4871

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="csj_char3265_mini_stack.yml")

    transducer = Transducer(
        cfgs=cfgs, global_batch_size=cfgs.serve.batch_size, setup_training=False
    )
    transducer._build()

    transducer.load_weights(cfgs.serve.model_path_from)

    transducer.save(filepath=cfgs.serve.model_path_to, include_optimizer=False)

    print("✨ Done.")
