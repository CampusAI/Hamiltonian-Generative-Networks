from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderNet(nn.Module):
    """Implementation of the encoder network, that encodes the input grayscale frames sequence
    into a latent space with the common variational reparametrization and sampling technique.
    """

    def __init__(self, seq_len):
        """Instantiate the convolutional layers

        Args:
            seq_len (int): Number of frames that compose a sequence.
        """
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels=seq_len, out_channels=32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.hidden_layers = [
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) for i in range(5)
        ]
        self.out_mean = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        self.out_logvar = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.conv1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        mean = self.out_mean(x)
        stdev = torch.exp(0.5 * self.out_logvar(x))
        epsilon = torch.randn_like(mean)
        x = mean + stdev * epsilon
        return x


class TransformerNet(nn.Module):
    """Implementation of the encoder-transformer network, that maps the latent space into the
    phase space.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
                               padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 64
    enc_net = EncoderNet(seq_len=SEQUENCE_LENGTH)

    rand_images = np.random.randint(0, 255, size=(BATCH_SIZE, SEQUENCE_LENGTH, 32, 32))
    rand_images_ts = torch.tensor(rand_images).float()

    z = enc_net(rand_images_ts)

    trans_net = TransformerNet(in_channels=48, out_channels=32)
    encoding = trans_net(z)
