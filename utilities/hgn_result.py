from utilities import conversions
from environments.environment import visualize_rollout
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HgnResult():
    """Class to bundle HGN guessed output information
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
        self.reconstructed_rollout = torch.empty(batch_shape).to(device)
        self.reconstruction_ptr = 0

    def set_input(self, rollout):
        """Store ground truth of system evolution

        Args:
            rollout (torch.Tensor): Tensor of shape (batch_size, seq_len, channels, height, width)
                containing the ground truth rollouts of a batch.
        """
        self.input = rollout

    def set_z(self, z_mean, z_logvar, z_sample):
        """Store latent encodings and correspondent distribution parameters.

        Args:
            z_mean (torch.Tensor): Batch of means of the latent distribution.
            z_logvar (torch.Tensor): Batch of log variances of the latent distributions.
            z_sample (torch.Tensor): Batch of latent encodings.
        """
        self.z_mean = z_mean
        self.z_logvar = z_logvar
        self.z_sample = z_sample

    def append_state(self, q, p):
        """Append the guessed position (q) and momentum (p) to guessed list 

        Args:
            q (torch.Tensor): Tensor with the abstract position.
            p (torch.Tensor): Tensor with the abstract momentum.
        """
        self.q_s.append(q)
        self.p_s.append(p)

    def append_reconstruction(self, reconstruction):
        """Append guessed reconstruction

        Args:
            reconstruction (torch.Tensor): Tensor of shape (seq_len, channels, height, width)
                containing the reconstructed rollout.
        """
        # TODO Fix this error. Workaround solution
        assert self.reconstruction_ptr < self.reconstructed_rollout.shape[1],\
            'Trying to add rollout number ' + str(self.reconstruction_ptr) + ' when batch has ' +\
            str(self.reconstructed_rollout.shape[0])
        self.reconstructed_rollout[:, self.reconstruction_ptr] = reconstruction
        self.reconstruction_ptr += 1

    def visualize(self):
        """Visualize the predicted rollout.
        """
        rollout_batch = conversions.to_channels_last(
            self.reconstructed_rollout).detach().numpy()
        sequence = conversions.batch_to_sequence(rollout_batch)
        visualize_rollout(sequence)
