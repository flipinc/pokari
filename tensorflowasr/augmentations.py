import tensorflow as tf

from spec_augment import SpectrogramAugmentation

AUGMENTATIONS = {}


class AugmentationExecutor:
    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    @tf.function
    def augment(self, inputs):
        outputs = inputs
        for au in self.augmentations:
            outputs = au.augment(outputs)
        return outputs


class Augmentation:
    def __init__(self, config: dict = None, use_tf: bool = False):
        if not config:
            config = {}

        self.before = self.tf_parse(config.pop("before", {}))
        # self.after = self.tf_parse(config.pop("after", {}))
        self.after = AugmentationExecutor([SpectrogramAugmentation()])

    @staticmethod
    def tf_parse(config: dict) -> list:
        augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(
                    f"No tf augmentation named: {key}\n"
                    f"Available tf augmentations: {AUGMENTATIONS.keys()}"
                )
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return AugmentationExecutor(augmentations)
