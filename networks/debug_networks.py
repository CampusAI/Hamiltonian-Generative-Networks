import torch
from inference_net import to_phase_space


class EncoderNet(torch.nn.Module):

    def __init__(self, phi=None, seq_len=2):
        super().__init__()
        self.seq_len = seq_len
        self.phi = phi if phi is not None else torch.tensor([1., 1.], requires_grad=True)

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
        return encoding, torch.zeros_like(encoding), torch.zeros_like(encoding)


class TransformerNet(torch.nn.Module):

    def __init__(self, w=None, sample=False):
        super().__init__()
        self.w = w if w is not None else torch.tensor([1., 1.], requires_grad=True)

    def forward(self, x):
        q_prime = self.w[0] * x[:, 0, :]
        p_prime = self.w[1] * x[:, 1, :]
        encoding = torch.stack((q_prime, p_prime), dim=1)
        return to_phase_space(encoding)


class HamiltonianNet(torch.nn.Module):

    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma if gamma is not None else torch.tensor([1., 1.], requires_grad=True)

    def forward(self, q, p):
        return self.gamma[0] * q + self.gamma[1] * p ** 2


class DecoderNet(torch.nn.Module):

    def __init__(self, theta=None):
        super().__init__()
        self.theta = theta if theta is not None else torch.tensor([2.])

    def forward(self, q):
        return self.theta * q



if __name__ == '__main__':
    inputs = torch.tensor([[[1.], [2.]], [[3.], [4.]]])

    encoder = EncoderNet()
    transformer = TransformerNet()
    hamiltonian = HamiltonianNet()
    decoder = DecoderNet()

    z, z_mean, z_var = encoder(inputs)
    q, p = transformer(z)
    h = hamiltonian(q, p)
    x_0 = decoder(q)
