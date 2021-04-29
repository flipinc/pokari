from hydra.experimental import compose, initialize
from hydra.utils import instantiate

if __name__ == "__main__":
    initialize(config_path="../configs/emformer", job_name="emformer")
    cfgs = compose(config_name="csj_char3265_mini_stack.yml")

    model = instantiate(
        cfgs.base_model,
        cfgs=cfgs,
        global_batch_size=2,
        setup_training=False,
    )
    model._build()

    model.load_weights(cfgs.trainer.model_path)

    model._save(filepath=cfgs.savedmodel_path)
