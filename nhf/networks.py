import os
import sys

import torch
import torch.nn as nn
import torch.distributions.multivariate_normal as multivariate_normal

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
        self.type(torch.float)

    def forward(self, q):
        # Compute mean
        mu = self.activation(self.mu_fc1(q))
        mu = self.activation(self.mu_fc2(mu))
        mu = self.activation(self.mu_fc3(mu))

        # Compute log-variance
        logvar = self.activation(self.logvar_fc1(q))
        logvar = self.activation(self.logvar_fc2(logvar))
        logvar = self.activation(self.logvar_fc3(logvar))

        # Reparametrization trick NOTE: I think it would be better to use normal.Normal.rsample()
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)
        # p = mu + std * epsilon
        # return p, mu, logvar
        std_mat = torch.diag_embed(std)
        mn = multivariate_normal.MultivariateNormal(mu, std_mat)
        p = mn.rsample(sample_shape=(q.size()[0], ))
        log_prob_p = mn.log_prob(p)
        return p, log_prob_p


class PartialEnergy(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activation = nn.Softplus()
        self.type(torch.float)

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
        self.integrator = Integrator(delta_t=None, method="Leapfrog")
        # Note change delta_t to delta_t/2 and apply two steps

    def forward(self, q, p, delta_t):
        q_next, p_next = self.integrator.step(q, p, self.hnn, delta_t)
        return q_next, p_next


class NHF:
    def __init__(self, input_size, delta_t, flow_steps, src_distribution):
        self.input_size = input_size
        self.delta_t = abs(delta_t)

        self.flows = [Flow(input_size) for _ in range(flow_steps)]
        self.encoder = Encoder(input_size)
        self.src_distribution = src_distribution

    def sample(self, delta_t=None):
        """Sample from the final distribution.

        Returns:
            (torch.Tensor): A sample from the learned distribution.
        """
        delta_t = self.delta_t
        assert delta_t > 0  # Integrator step must be positive when sampling
        sample_shape = (1,)
        q = self.src_distribution.sample(
            sample_shape=sample_shape
        )  #NOTE: Not sure if should sample them separately or together
        p = self.src_distribution.sample(sample_shape=sample_shape)
        q.requires_grad_(True)
        p.requires_grad_(True)
        
        for flow in self.flows:
            q, p = flow(q=q, p=p, delta_t=delta_t)
        return q

    def inference(self, q):
        """Get log_prob of point q from learned distribution.

        Returns:
            (float): Learned distribution log_prob of sample q.
        """
        delta_t = -self.delta_t
        assert delta_t < 0  # Integrator step must be negative when infering

        # Get p_T from q_T
        _, p, _ = self.encoder(
            q)  #NOTE: Should we use the mean when doing inference?

        # Apply the inverse of all the flows
        for flow in reversed(self.flows):
            q, p = flow(q=q, p=p, delta_t=delta_t)
        return self.src_distribution.log_prob(q).item()

    def elbo(self, q, lagrange_multiplier=1.0):
        delta_t = -self.delta_t
        assert delta_t < 0  # Integrator step must be negative when infering

        # Get p_T from q_T
        p, log_prob_p = self.encoder(q)

        # Apply the inverse of all the flows
        for flow in reversed(self.flows):
            q, p = flow(q=q, p=p, delta_t=delta_t)

        # Compute ELBO(q_T)
        elbo_q = self.src_distribution.log_prob(q) +\
                 self.src_distribution.log_prob(p) -\
                 lagrange_multiplier*log_prob_p
        return torch.mean(elbo_q)
