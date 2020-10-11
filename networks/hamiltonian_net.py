import torch
from torch import nn


class HamiltonianNet(nn.Module):

    def __init__(self, in_channels, dtype=torch.float):
        """Create the layers of the Hamiltonian network.

        Args:
            in_channels (int): Number of input channels.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        self.num_flat_features = 64 * 4 * 4
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=self.num_flat_features, out_features=1)
        self.type(dtype)

    def forward(self, q, p):
        """Forward pass that returns the Hamiltonian for the given q and p inputs.

        q and p must be two N x C x H x W tensors, where N is the batch size, C the number of
        channels, H and W the height and width.

        Args:
            q (torch.Tensor): The tensor corresponding to the position in abstract space.
            p (torch.Tensor): The tensor corresponding to the momentum in abstract space.

        Returns:
            A N x 1 tensor with the Hamiltonian for each input in the batch.
        """
        x = torch.cat((q, p), dim=1)  # Concatenate q and p to obtain a N x 2C x H x W tensor
        x = self.in_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out_conv(x)
        x = x.view(-1, self.num_flat_features)
        x = self.linear(x)
        return x
