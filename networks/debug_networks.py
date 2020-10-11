import torch
from torch import nn
from .inference_net import to_phase_space


class EncoderNet(nn.Module):

    def __init__(self, phi=None, seq_len=2, dtype=torch.float):
        super().__init__()
        self.seq_len = seq_len
        self.phi = phi if phi is not None else nn.Parameter(
            torch.tensor([1., 2.], requires_grad=True, dtype=dtype)
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor) must be a N x k x shape where N is the batch size, k is the length
                of the sequence, and shape is the shape of each frame in the sequence. Here we
                assume k=2 and shape = 1. The encoding is given by
                [phi[0] * x[0], phi[1] * (x[1] - x[0])]
        Returns:
            A N x 2 encoding, where is N the batch size, and each 2-dimensional element is [q, p]
        """
        q = self.phi[0] * x[:, 0, :]
        p = self.phi[1] * (x[:, 1, :] - x[:, 0, :])
        encoding = torch.stack((q, p), dim=1)
        return encoding, torch.zeros_like(encoding), torch.ones_like(encoding)


class TransformerNet(torch.nn.Module):

    def __init__(self, w=None, sample=False, dtype=torch.float):
        super().__init__()
        self.w = w if w is not None else nn.Parameter(
            torch.tensor([1., 1.], requires_grad=True, dtype=dtype)
        )

    def forward(self, x):
        q_prime = self.w[0] * x[:, 0, :]
        p_prime = self.w[1] * x[:, 1, :]
        encoding = torch.stack((q_prime, p_prime), dim=1)
        return to_phase_space(encoding)


class HamiltonianNet(torch.nn.Module):

    def __init__(self, gamma=None, dtype=torch.float):
        super().__init__()
        self.gamma = gamma if gamma is not None else nn.Parameter(
            torch.tensor([3., 4.], requires_grad=True, dtype=dtype)
        )

    def forward(self, q, p):
        return self.gamma[0] * q + self.gamma[1] * p ** 2


class DecoderNet(torch.nn.Module):

    def __init__(self, theta=None, dtype=torch.float):
        super().__init__()
        self.theta = theta if theta is not None else nn.Parameter(torch.tensor([2.], dtype=dtype))

    def forward(self, q):
        return self.theta * q

