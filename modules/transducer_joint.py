from typing import List, Optional, Union

import contextmanager
import torch
import torch.nn as nn


class RNNTJoint(nn.Module):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        num_classes: int, specifying the vocabulary size that the joint network
            must predict, excluding the RNNT blank token.
    """

    def __init__(
        self, enc_hidden: int, pred_hidden: int, joint_hidden: int, num_classes: int,
    ):
        super().__init__()

        self._vocab_size = num_classes
        self._num_classes = num_classes + 1  # add 1 for blank symbol

        self._loss = None
        self._wer = None

        # Required arguments
        self.enc_hidden = enc_hidden
        self.pred_hidden = pred_hidden
        self.joint_hidden = joint_hidden

        self.pred, self.enc, self.joint_net = self._joint_net(
            num_classes=self._num_classes,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.enc_hidden,
            joint_n_hidden=self.joint_hidden,
        )

    def forward(
        self, encoder_outputs: torch.Tensor, decoder_outputs: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U)
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
        return out

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original
            paper.

            The original paper proposes the following steps :
            (enc, dec)
            -> Expand + Concat + Sum [B, T, U, H1+H2]
            -> Forward through joint hidden [B, T, U, H]
            -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and
            joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2)
            -> Sum [B, T, U, H]
            -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        # f = [B, T, H1]
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        del f, g

        res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if not res.is_cuda:  # Use log softmax only if on CPU
            res = res.log_softmax(dim=-1)

        return res

    def _joint_net(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding
                the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        activation = torch.nn.ReLU(inplace=True)

        layers = [activation] + [torch.nn.Linear(joint_n_hidden, num_classes)]
        return pred, enc, torch.nn.Sequential(*layers)

    def freeze(self) -> None:
        r"""
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
