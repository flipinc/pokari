import contextmanager
import torch.nn as nn


class BaseModule(nn.Module):
    def freeze(self) -> None:
        """
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes a module, yields control and finally
        unfreezes the module.
        """
        self.freeze()

        try:
            yield
        finally:
            self.unfreeze()
