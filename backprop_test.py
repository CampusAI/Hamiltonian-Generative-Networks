import torch
import sys
from pathlib import Path
from utils import integrator


DELTA_T = 0.1


class Encoder(torch.nn.Module):

    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def forward(self, frames):
        return self.phi[0] * frames[0], self.phi[1] * (frames[1] - frames[0]) / DELTA_T


class HamNet(torch.nn.Module):

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, q, p):
        return gamma[0] * q + gamma[1] * p ** 2


class Decoder(torch.nn.Module):

    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def forward(self, q):
        return self.theta * q


if __name__ == '__main__':

    # Parameters
    phi = torch.tensor([1., 1.], requires_grad=True)
    gamma = torch.tensor([1., 1.], requires_grad=True)
    theta = torch.tensor([2.], requires_grad=True)
    opt = torch.optim.SGD([theta, gamma, phi], lr=0.01)

    # Input data
    frames = torch.tensor([1., 2.], requires_grad=True)

    # Encoder
    encoder = Encoder(phi=phi)

    # Hamiltonian net
    hamiltonian_net = HamNet(gamma)

    # Integrator
    integ = integrator.Integrator(DELTA_T)

    # Decoder
    decoder = Decoder(theta)

    # Forward pass
    q_0, p_0 = encoder(frames)
    print('Encoded t=0 q: ' + str(q_0) + ' p: ' + str(p_0))

    x_0 = decoder(q_0)  # First frame
    print('Decoded t=0 q_0:' +str(x_0))

    q_1, p_1 = integ.step(q_0, p_0, hamiltonian_net)
    print('Rollouts t=1 q: ' + str(q_1) + ' p: ' + str(p_1))

    x_1 = decoder(q_1)  # Second frame

    loss = (x_0 - frames[0]) ** 2 + (x_1 - frames[1]) ** 2

    loss.backward()
    print('T=0 Gradient wrt theta: ' + str(theta.grad))
    print('T=0 Gradient wrt gamma: ' + str(gamma.grad))
    print('T=0 Gradient wrt phi: ' + str(phi.grad))

    opt.zero_grad()


