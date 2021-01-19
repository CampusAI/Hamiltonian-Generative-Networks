from utilities import conversions
from environments.environment import visualize_rollout
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HgnResult():
    """Class to bundle HGN guessed output information.
    """

    def __init__(self, batch_shape, device):
        """Instantiate the HgnResult that will contain all the information of the forward pass
        over a single batch of rollouts.

        Args:
            batch_shape (torch.Size): Shape of a batch of reconstructed rollouts, returned by
                batch.shape.
            device (str): String with the device to use. E.g. 'cuda:0', 'cpu'.
        """
        self.input = None
        self.z_mean = None
        self.z_logvar = None
        self.z_sample = None
        self.q_s = []
        self.p_s = []
        self.energies = []  # Estimated energy of the system by the Hamiltonian network
        self.reconstructed_rollout = torch.empty(batch_shape).to(device)
        self.reconstruction_ptr = 0

    def set_input(self, rollout):
        """Store ground truth of system evolution.

        Args:
            rollout (torch.Tensor): Tensor of shape (batch_size, seq_len, channels, height, width)
                containing the ground truth rollouts of a batch.
        """
        self.input = rollout

    def set_z(self, z_sample, z_mean=None, z_logvar=None):
        """Store latent encodings and correspondent distribution parameters.

        Args:
            z_sample (torch.Tensor): Batch of latent encodings.
            z_mean (torch.Tensor, optional): Batch of means of the latent distribution.
            z_logvar (torch.Tensor, optional): Batch of log variances of the latent distributions.
        """
        self.z_mean = z_mean
        self.z_logvar = z_logvar
        self.z_sample = z_sample

    def append_state(self, q, p):
        """Append the guessed position (q) and momentum (p) to guessed list .

        Args:
            q (torch.Tensor): Tensor with the abstract position.
            p (torch.Tensor): Tensor with the abstract momentum.
        """
        self.q_s.append(q)
        self.p_s.append(p)

    def append_reconstruction(self, reconstruction):
        """Append guessed reconstruction.

        Args:
            reconstruction (torch.Tensor): Tensor of shape (seq_len, channels, height, width).
                containing the reconstructed rollout.
        """
        assert self.reconstruction_ptr < self.reconstructed_rollout.shape[1],\
            'Trying to add rollout number ' + str(self.reconstruction_ptr) + ' when batch has ' +\
            str(self.reconstructed_rollout.shape[0])
        self.reconstructed_rollout[:, self.reconstruction_ptr] = reconstruction
        self.reconstruction_ptr += 1

    def append_energy(self, energy):
        """Append the guessed system energy to energy list.

        Args:
            energy (torch.Tensor): Energy of each trajectory in the batch.
        """
        self.energies.append(energy)

    def get_energy(self):
        """Get the average energy of that rollout and the average of each trajectory std.

        Returns:
            (tuple(float, float)): (average_energy, average_std_energy) average_std_energy is computed as follows:
            For each trajectory in the rollout, compute the std of the energy and average across trajectories.
        """
        energies = np.array(self.energies) 
        energy_std = np.std(energies, axis=0)
        return np.mean(energies), np.mean(energy_std)

    def visualize(self, interval=50, show_step=False):
        """Visualize the predicted rollout.
        """
        rollout_batch = conversions.to_channels_last(
            self.reconstructed_rollout).detach().cpu().numpy()
        sequence = conversions.batch_to_sequence(rollout_batch)
        visualize_rollout(sequence, interval=interval, show_step=show_step)
