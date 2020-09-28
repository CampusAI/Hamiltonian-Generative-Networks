from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InferenceNet(nn.Module):

    def __init__(self, seq_len):
        """Create the inference network, that encodes the input frames sequence into a latent space

        Args:
            seq_len (int): Number of frames that compose a sequence.
        """
        super().__init__()
        # TODO: What kernel size?
        self.input_conv = nn.Conv2d(in_channels=seq_len, out_channels=32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.hidden_layers = [
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) for i in range(5)
        ]
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        # Up to here we have the latent encoding to train with reparam-trick

    def forward(self, x):
        x = self.input_conv(x)
        x = self.conv1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 64
    inf_net = InferenceNet(seq_len=SEQUENCE_LENGTH)

    rand_images = np.random.randint(0, 255, size=(BATCH_SIZE, SEQUENCE_LENGTH, 32, 32))
    rand_images_ts = torch.tensor(rand_images).float()

    z = inf_net(rand_images_ts)