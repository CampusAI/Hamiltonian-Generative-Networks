import sys
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments import pendulum
from utilities import integrator


def plot(theta, theta_dot, real_theta, real_theta_dot):
    t = np.arange(theta.shape[0])
    plt.plot(t, theta, label='Theta')
    plt.plot(t, theta_dot, label='Theta_dot')
    plt.plot(t, real_theta, label='Real Theta')
    plt.plot(t, real_theta_dot, label='Real Theta_dot')
    plt.legend()
    plt.show()


class Hamiltonian(torch.nn.Module):
    """Compute the hamiltonian for the given angle and angular velocity
    """

    def __init__(self, m, l, g):
        super().__init__()
        self.m = torch.nn.Parameter(torch.tensor(m, requires_grad=True))
        self.l = torch.nn.Parameter(torch.tensor(l, requires_grad=True))
        self.g = torch.nn.Parameter(torch.tensor(g, requires_grad=True))

    def forward(self, q, p):
        return p**2 / (2 * self.m * self.l**2) + self.m * self.g * self.l * (1 - torch.cos(q))


if __name__ == '__main__':
    num_frames = 100
    delta_time = 0.1
    m = 1.
    l = 1.
    g = 1.

    # q is theta, p is theta_dot
    q_0, p_0 = [0.], [1.]

    _env = pendulum.Pendulum(mass=m, length=l, g=g, q=q_0, p=p_0)
    _env._evolution(total_time=num_frames * delta_time, delta_time=delta_time)

    _theta, _theta_dot = _env._rollout

    # To tensors
    q_0_ts = torch.tensor(q_0, requires_grad=True)
    p_0_ts = torch.tensor(p_0, requires_grad=True)

    hnn = Hamiltonian(m=m, l=l, g=g)
    leapfrog = integrator.Integrator(delta_t=delta_time, method='Euler')
    opt = torch.optim.Adam(hnn.parameters(), lr=0.1)

    # Initialize empty tensors to store q and p at each timestep
    q = torch.empty(num_frames)
    p = torch.empty(num_frames)
    q[0] = q_0_ts
    p[0] = p_0_ts
    for i in range(1, num_frames):
        q_next, p_next = leapfrog.step(q[i - 1], p[i - 1], hnn)
        q[i] = q_next
        p[i] = p_next
    q = q.detach().numpy()
    p = p.detach().numpy()

    print(q.shape)
    print(p.shape)
    print(_theta.shape)
    print(_theta_dot.shape)

    plot(q, p, _theta, _theta_dot)