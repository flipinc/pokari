import os
import warnings

import tensorflow as tf
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from model import StreamingTransducer
from slice_dataset import SliceDataset
from speech import SpeechFeaturizer
from text import SubwordFeaturizer

warnings.simplefilter("ignore")
tf.get_logger().setLevel("ERROR")

tf.keras.backend.clear_session()

initialize(config_path="../configs/emformer", job_name="emformer")
args = compose(config_name="config.yml")

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    visible_gpus = [gpus[i] for i in args.devices]
    tf.config.set_visible_devices(visible_gpus, "GPU")
    print("Run on", len(visible_gpus), "Physical GPUs")

strategy = tf.distribute.MirroredStrategy()

speech_featurizer = SpeechFeaturizer(OmegaConf.to_container(args.speech_config))

if args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(
        OmegaConf.to_container(args.decoder_config),
        args.subwords,
    )
else:
    print("Generating subwords ...")
    text_featurizer = SubwordFeaturizer.build_from_corpus(
        OmegaConf.to_container(args.decoder_config),
        args.subwords_corpus,
    )
    text_featurizer.save_to_file(args.subwords)


train_dataset = SliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    **OmegaConf.to_container(args.learning_config.train_dataset_config)
)

eval_dataset = SliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    **OmegaConf.to_container(args.learning_config.eval_dataset_config)
)

with strategy.scope():
    global_batch_size = args.learning_config.running_config.batch_size
    global_batch_size *= strategy.num_replicas_in_sync
    # build model
    streaming_transducer = StreamingTransducer(
        **OmegaConf.to_container(args.model_config),
        vocabulary_size=text_featurizer.num_classes
    )
    streaming_transducer._build(speech_featurizer.shape)
    # streaming_transducer.summary(line_length=150)

    optimizer = tf.keras.optimizers.get(
        OmegaConf.to_container(args.learning_config.optimizer_config)
    )

    streaming_transducer.compile(
        optimizer=optimizer,
        global_batch_size=global_batch_size,
        blank=text_featurizer.blank,
    )

    train_data_loader = train_dataset.create(global_batch_size)
    eval_data_loader = eval_dataset.create(global_batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            **OmegaConf.to_container(args.learning_config.running_config.checkpoint)
        ),
        tf.keras.callbacks.experimental.BackupAndRestore(
            args.learning_config.running_config.states_dir
        ),
        tf.keras.callbacks.TensorBoard(
            **OmegaConf.to_container(args.learning_config.running_config.tensorboard)
        ),
    ]

    streaming_transducer.fit(
        train_data_loader,
        epochs=args.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
    )
