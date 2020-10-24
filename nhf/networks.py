import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.integrator import Integrator

class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Mean network
        self.mu_fc1 = nn.Linear(input_size, 128)
        self.mu_fc2 = nn.Linear(128, 128)
        self.mu_fc3 = nn.Linear(128, input_size)

        # Std network
        self.logvar_fc1 = nn.Linear(input_size, 128)
        self.logvar_fc2 = nn.Linear(128, 128)
        self.logvar_fc3 = nn.Linear(128, input_size)
    
        # Activation
        self.activation = nn.ReLU()

    def forward(self, q):
        # Compute mean
        mu = self.activation(self.mu_fc1(q))
        mu = self.activation(self.mu_fc2(mu))
        mu = self.activation(self.mu_fc3(mu))

        # Compute log-variance
        logvar = self.activation(self.logvar_fc1(q))
        logvar = self.activation(self.logvar_fc2(logvar))
        logvar = self.activation(self.logvar_fc3(logvar))

        # Reparametrization trick
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mu)
        p = mu + std * epsilon
        return p, mu, logvar


class PartialEnergy(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activation = nn.Softplus()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x


class Hamiltonian(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.kinetic_energy = PartialEnergy(input_size)
        self.potential_energy = PartialEnergy(input_size)

    def forward(self, q, p):
        energy = self.potential_energy(q) + self.kinetic_energy(p)
        return energy


class Flow(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hnn = Hamiltonian(input_size)
        self.integrator = Integrator(delta_t=0.125, method="Leapfrog")
        # Note change delta_t to delta_t/2 and apply two steps

    def forward(self, s):
        pass


# class NHF:
#     def __init__(self, input_size, rollout_length):
#         self.flows = [Flow(input_size)]*rollout_length
#         self.encoder = Encoder(input_size)

#     def forward(self, rollout):

