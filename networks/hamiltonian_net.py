import torch
from torch import nn


class HamiltonianNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.num_flat_features = 64 * 4 * 4
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=self.num_flat_features, out_features=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out_conv(x)
        x = x.view(-1, self.num_flat_features)
        x = self.linear(x)
        return x