"""This module contains simple, easy to debug networks that have the same interface as the other
networks in the package.
"""

import sys
import os

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks import decoder_net
from networks import encoder_net
from networks import hamiltonian_net
from networks import transformer_net


class EncoderNet(encoder_net.EncoderNet):
    """Network that encodes the input value into a two-dimensional latent encoding.
    """
    def __init__(self, phi=None, dtype=torch.float):
        """Create the encoder network.

        Args:
            phi (torch.Tensor): 2-dimensional Tensor with weights.
            dtype (torch.dtype): Type of the weights.
        """
        nn.Module.__init__(self)
        self.phi = phi if phi is not None else nn.Parameter(
            torch.tensor([1., 2.], requires_grad=True, dtype=dtype))

    def forward(self, x):
        """Compute the encoding of x.

        Args:
            x (torch.Tensor) must be a N x k x shape where N is the batch size, k is the length
                of the sequence, and shape is the shape of each frame in the sequence. Here we
                assume k=2 and shape = 1. The encoding is given by
                [phi[0] * x[0], phi[1] * (x[1] - x[0])]
        Returns:
            A tuple (encoding, mean, var), where encoding is a N x 2 Tensor, N is the batch
            size. mean and var are the mean and variance tensors, returned for compatibility.
        """
        q = self.phi[0] * x[:, 0, :]
        p = self.phi[1] * (x[:, 1, :] - x[:, 0, :])
        encoding = torch.stack((q, p), dim=1)
        return encoding, torch.zeros_like(encoding), torch.ones_like(encoding)


class TransformerNet(transformer_net.TransformerNet):
    """Transforms the given encoding into abstract phase space q and p.
    """
    def __init__(self, w=None, dtype=torch.float):
        """Create the transformer net by setting the weighs.

        Args:
            w (torch.Tensor): Tensor of weights.
            dtype (torch.dtype): Type of the weights.
        """
        nn.Module.__init__(self)
        self.w = w if w is not None else nn.Parameter(
            torch.tensor([1., 1.], requires_grad=True, dtype=dtype))

    def forward(self, x):
        """Transform the two dimensional input tensor x into q, p as q = w_0 * x_0, p = w_1 * x_1

        Args:
            x (torch.Tensor): Two dimensional latent encoding.

        Returns:
            A tuple two Tensors q and p.
        """
        q = self.w[0] * x[:, 0, :]
        p = self.w[1] * x[:, 1, :]
        encoding = torch.stack((q, p), dim=1)
        return self.to_phase_space(encoding)


class HamiltonianNet(hamiltonian_net.HamiltonianNet):
    """Computes the hamiltonian from the given q and p.
    """
    def __init__(self, gamma=None, dtype=torch.float):
        """Create the hamiltonian net.

        Args:
            gamma (torch.Tensor): A two dimensional tensor with the weights.
            dtype (torch.type): Type of the weights.
        """
        nn.Module.__init__(self)
        self.gamma = gamma if gamma is not None else nn.Parameter(
            torch.tensor([3., 4.], requires_grad=True, dtype=dtype))

    def forward(self, q, p):
        return self.gamma[0] * q + self.gamma[1] * p**2


class DecoderNet(decoder_net.DecoderNet):
    """Decoder net debug implementation, where the input q is decoded as q * theta
    """
    def __init__(self, theta=None, dtype=torch.float):
        """Create the debug decoder network.

        Args:
            theta (torch.Tensor): 1 dimensional Tensor containing theta.
            dtype (torch.dtype): Type of theta.
        """
        nn.Module.__init__(self)
        self.theta = theta if theta is not None else nn.Parameter(
            torch.tensor([2.], dtype=dtype))

    def forward(self, q):
        """Returns q * theta.

        Args:
            q (torch.tensor): A one-dimensional Tensor.

        Returns:
            A one dimensional Tensor.
        """
        return self.theta * q


if __name__ == '__main__':
    enc = EncoderNet()
    transf = TransformerNet()
    ham = HamiltonianNet()
    dec = DecoderNet()