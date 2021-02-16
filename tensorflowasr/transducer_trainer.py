import os
import warnings

import tensorflow as tf
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from audio_featurizer import AudioFeaturizer
from dataset import Dataset
from text_featurizer import SubwordFeaturizer
from transducer import Transducer

warnings.simplefilter("ignore")
tf.get_logger().setLevel("ERROR")

tf.keras.backend.clear_session()

initialize(config_path="../configs/emformer", job_name="emformer")
cfgs = compose(config_name="config.yml")

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": cfgs.mxp})

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    visible_gpus = [gpus[i] for i in cfgs.devices]
    tf.config.set_visible_devices(visible_gpus, "GPU")
    print("Run on", len(visible_gpus), "Physical GPUs")

strategy = tf.distribute.MirroredStrategy()

audio_featurizer = AudioFeaturizer(**OmegaConf.to_container(cfgs.speech_config))

if cfgs.subwords and os.path.exists(cfgs.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(
        OmegaConf.to_container(cfgs.decoder_config),
        cfgs.subwords,
    )
else:
    print("Generating subwords ...")
    text_featurizer = SubwordFeaturizer.build_from_corpus(
        OmegaConf.to_container(cfgs.decoder_config),
        cfgs.subwords_corpus,
    )
    text_featurizer.save_to_file(cfgs.subwords)

train_dataset = Dataset(
    speech_featurizer=audio_featurizer,
    text_featurizer=text_featurizer,
    **OmegaConf.to_container(cfgs.learning_config.train_dataset_config),
)

eval_dataset = Dataset(
    speech_featurizer=audio_featurizer,
    text_featurizer=text_featurizer,
    **OmegaConf.to_container(cfgs.learning_config.eval_dataset_config),
)

with strategy.scope():
    global_batch_size = cfgs.learning_config.running_config.batch_size
    global_batch_size *= strategy.num_replicas_in_sync

    transducer = Transducer(
        vocab_size=text_featurizer.num_classes,
        audio_featurizer=audio_featurizer,
        cfgs=cfgs,
    )
    transducer._build(audio_featurizer.shape)
    transducer.summary(line_length=150)

    optimizer = tf.keras.optimizers.get(
        OmegaConf.to_container(cfgs.learning_config.optimizer_config)
    )

    transducer.compile(
        optimizer=optimizer,
        global_batch_size=global_batch_size,
        blank=text_featurizer.blank,
    )

    train_data_loader = train_dataset.create(global_batch_size)
    eval_data_loader = eval_dataset.create(global_batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            **OmegaConf.to_container(cfgs.learning_config.running_config.checkpoint)
        ),
        tf.keras.callbacks.experimental.BackupAndRestore(
            cfgs.learning_config.running_config.states_dir
        ),
        tf.keras.callbacks.TensorBoard(
            **OmegaConf.to_container(cfgs.learning_config.running_config.tensorboard)
        ),
    ]

    transducer.fit(
        train_data_loader,
        epochs=cfgs.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
    )
