import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.environments import visualize_rollout


class HgnResult():
    """Class to bundle HGN guessed output information
    """
    def __init__(self):
        self.input = None
        self.z_mean = None
        self.z_logvar = None
        self.z_sample = None
        self.q_s = []
        self.p_s = []
        self.reconstructed_rollout = None

    def set_input(self, rollout):
        """Store ground truth of system evolution

        Args:
            rollout (torch.Tensor): Ground truth of the system evolution, concatenated along last axis
        """
        self.input = rollout

    def set_z(self, z_mean, z_logvar, z_sample):
        """Store latent variable conditions

        Args:
            z_mean (torch.Tensor): Mean of q_z
            z_logvar (torch.Tensor): Standard dev of q_z
            z_sample (torch.Tensor): Sample taken from q_z distribution
        """
        self.z_mean = z_mean
        self.z_logvar = z_logvar
        self.z_sample = z_sample

    def append_state(self, q, p):
        """Append the guessed position (q) and momentum (p) to guessed list 

        Args:
            q (torch.Tensor): Position encoded information
            p (torch.Tensor): Momentum encoded information
        """
        self.q_s.append(q)
        self.p_s.append(p)

    def append_reconstruction(self, reconstruction):
        """Append guessed reconstruction

        Args:
            reconstruction (torch.Tensor): Decoder of the HGN output
        """
        if self.reconstructed_rollout is None:
            self.reconstructed_rollout = reconstruction
        else:
            self.reconstructed_rollout = torch.cat(
                (self.reconstructed_rollout, reconstruction), dim=1)

    def visualize(self):
        rollout = self.reconstructed_rollout.detach().numpy()
        rollout = np.squeeze(rollout, axis=0)
        rollout = np.array(np.split(rollout, len(self.q_s), axis=0))
        print(rollout.shape)
        rollout = rollout.transpose((0, 2, 3, 1))

        plt.hist(rollout.flatten())
        plt.show()

        rollout = np.array(250*rollout, dtype=np.uint8)

        # plt.hist(rollout.flatten())

        print(rollout.shape)
        print(rollout)
        visualize_rollout(rollout)
