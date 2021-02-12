import torch
import torch.nn as nn


class TransducerJoint(nn.Module):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        num_classes: int, specifying the vocabulary size that the joint network
            must predict, excluding the RNNT blank token.
    """

    def __init__(
        self,
        encoder_hidden: int,
        predictor_hidden: int,
        dim_model: int,
        vocab_size: int,
        activation: str,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._num_classes = vocab_size + 1  # add 1 for blank symbol

        self.linear_encoder = torch.nn.Linear(encoder_hidden, dim_model)
        self.linear_predictor = torch.nn.Linear(predictor_hidden, dim_model)

        activation = activation.lower()

        if activation == "relu":
            activation = torch.nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            activation = torch.nn.Tanh()

        layers = [activation] + [torch.nn.Linear(dim_model, self._num_classes)]
        self.joint_net = torch.nn.Sequential(*layers)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        predictor_outputs: torch.Tensor,
    ) -> torch.Tensor:
        # encoder = (B, D, T)
        # decoder = (B, D, U)
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        predictor_outputs = predictor_outputs.transpose(1, 2)  # (B, U, D)

        out = self.joint(encoder_outputs, predictor_outputs)  # [B, T, U, V + 1]
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
        f = self.linear_encoder(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.linear_predictor(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        # torch script does not support del with multiples targets
        del f
        del g

        res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if not res.is_cuda:  # Use log softmax only if on CPU
            res = res.log_softmax(dim=-1)

        return res
