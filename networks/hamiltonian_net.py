"""This module contains an implementation of the Hamiltonian network described in the paper.
The Hamiltonian network takes the abstract positions and momenta, q and p, and computes a scalar
value that is interpreted as the Hamiltonian.
"""

import torch
from torch import nn
from torch.nn import functional as F


class HamiltonianNet(nn.Module):
    """The Hamiltonian network, composed of 6 convolutional layers and a final linear layer.
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 6,
        'n_filters': [32, 64, 64, 64, 64, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3],
        'strides': [1, 1, 1, 1, 1, 1, 1, 1],
    }

    def __init__(self,
                 in_shape,
                 hidden_conv_layers=None,
                 n_filters=None,
                 kernel_sizes=None,
                 strides=None,
                 paddings=None,
                 act_func=nn.Softplus(),
                 dtype=torch.float):
        """Create the layers of the Hamiltonian network.

        If K is the total number of convolutional layers, then hidden_conv_layers = K - 2.
        The length of n_filters, kernel_sizes, and strides must be K. If all of them are None,
        HamiltonianNet.DEFAULT_PARAMS will be used.

        Args:
            in_shape (tuple): Shape of input elements (channels, height, width).
            hidden_conv_layers (int): Number of hidden convolutional layers (excluding the input and
                the two output layers for mean and variance).
            n_filters (list): List with number of filters for each of the hidden layers.
            kernel_sizes (list): List with kernel sizes for each convolutional layer.
            strides (list): List with strides for each convolutional layer.
            act_func (torch.nn.Module): The activation function to apply after each layer.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        if all(var is None for var in (hidden_conv_layers, n_filters,
                                       kernel_sizes, strides, paddings)):
            hidden_conv_layers = HamiltonianNet.DEFAULT_PARAMS[
                'hidden_conv_layers']
            n_filters = HamiltonianNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = HamiltonianNet.DEFAULT_PARAMS['kernel_sizes']
            strides = HamiltonianNet.DEFAULT_PARAMS['strides']
            paddings = HamiltonianNet.DEFAULT_PARAMS['paddings']
        elif all(var is not None for var in (hidden_conv_layers, n_filters,
                                             kernel_sizes, strides, paddings)):
            # If no Nones, check consistency
            assert len(n_filters) == hidden_conv_layers + 1,\
                'n_filters must be a list of length hidden_conv_layers + 2 ' \
                '(' + str(hidden_conv_layers + 2) + ' in this case).'
            assert len(kernel_sizes) == hidden_conv_layers + 2 and \
                   len(strides) == len(kernel_sizes) and \
                   len(paddings) == len(kernel_sizes), \
                   'kernel_sizes and strides must be lists with values foreach layer in the ' \
                   'network (' + str(hidden_conv_layers + 2) + ' in this case).'
        else:
            raise ValueError(
                'Args hidden_conv_layers, n_filters, kernel_sizes, and strides'
                'can only be either all None, or all defined by the user.'
            )
        self.paddings = paddings
        conv_paddings = [0 if isinstance(p, list) else p for p in paddings]
        in_channels = in_shape[0] * 2
        self.in_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=n_filters[0],
                                 kernel_size=kernel_sizes[0],
                                 padding=conv_paddings[0],
                                 stride=strides[0])
        self.hidden_layers = nn.ModuleList(modules=[
            nn.Conv2d(in_channels=n_filters[i],
                      out_channels=n_filters[i + 1],
                      kernel_size=kernel_sizes[i + 1],
                      padding=conv_paddings[i + 1],
                      stride=strides[i + 1]) for i in range(hidden_conv_layers)
        ])
        self.out_conv = nn.Conv2d(in_channels=n_filters[-1],
                                  out_channels=1,
                                  kernel_size=2,
                                  padding=0)
        self.activation = act_func
        self.type(dtype)

    def forward(self, q, p):
        """Forward pass that returns the Hamiltonian for the given q and p inputs.

        q and p must be two (batch_size, channels, height, width) tensors.

        Args:
            q (torch.Tensor): The tensor corresponding to the position in abstract space.
            p (torch.Tensor): The tensor corresponding to the momentum in abstract space.

        Returns:
            A (batch_size, 1) shaped tensor with the energy for each input in the batch.
        """
        x = torch.cat(
            (q, p),
            dim=1)  # Concatenate q and p to obtain a N x 2C x H x W tensor
        if isinstance(self.paddings[0], list):
            x = F.pad(x, self.paddings[0])
        x = self.activation(self.in_conv(x))
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(self.paddings[i + 1], list):
                x = F.pad(x, self.paddings[i + 1])
            x = self.activation(layer(x))
        x = self.activation(self.out_conv(x))
        x = x.squeeze(dim=1).squeeze(dim=1)
        return x


if __name__ == '__main__':
    hamiltonian_net = HamiltonianNet(in_shape=(16, 4, 4))
    q, p = torch.randn((2, 128, 16, 4, 4))
    h = hamiltonian_net(q, p)
