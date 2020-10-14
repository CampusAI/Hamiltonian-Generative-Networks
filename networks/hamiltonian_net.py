"""This module contains an implementation of the Hamiltonian network described in the paper.
The Hamiltonian network takes the abstract positions and momenta, q and p, and computes a scalar
value that is interpreted as the Hamiltonian.
"""

import torch
from torch import nn


class HamiltonianNet(nn.Module):
    """The Hamiltonian network, composed of 6 convolutional layers and a final linear layer.
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 6,
        'n_filters': [32, 64, 64, 64, 64, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'strides': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }

    def __init__(self, in_shape, hidden_conv_layers=None, n_filters=None, kernel_sizes=None,
                 strides=None, act_func=nn.ReLU(), dtype=torch.float):
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
        if all(var is None for var in (hidden_conv_layers, n_filters, kernel_sizes, strides)):
            hidden_conv_layers = HamiltonianNet.DEFAULT_PARAMS['hidden_conv_layers']
            n_filters = HamiltonianNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = HamiltonianNet.DEFAULT_PARAMS['kernel_sizes']
            strides = HamiltonianNet.DEFAULT_PARAMS['strides']
        elif all(var is not None for var in (hidden_conv_layers, n_filters, kernel_sizes, strides)):
            # If no Nones, check consistency
            assert len(n_filters) == hidden_conv_layers + 2,\
                'n_filters must be a list of length hidden_conv_layers + 2 ' \
                '(' + str(hidden_conv_layers + 2) + ' in this case).'
            assert len(kernel_sizes) == hidden_conv_layers + 2 and \
                   len(strides) == hidden_conv_layers + 2, \
                   'kernel_sizes and strides must be lists with values foreach layer in the ' \
                   'network (' + str(hidden_conv_layers + 2) + ' in this case).'
        else:
            raise ValueError('Args hidden_conv_layers, n_filters, kernel_sizes, and strides'
                             'can only be either all None, or all defined by the user.')
        in_channels = in_shape[0] * 2
        paddings = [int(k/2) for k in kernel_sizes]
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters[0],
            kernel_size=kernel_sizes[0],
            padding=paddings[0])
        out_size = int((in_shape[1] - kernel_sizes[0] + 2 * paddings[0])) / strides[0] + 1
        self.hidden_layers = nn.ModuleList(
            modules=[
                nn.Conv2d(
                    in_channels=n_filters[i],
                    out_channels=n_filters[i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    padding=paddings[i + 1],
                    stride=strides[i + 1]
                )
                for i in range(hidden_conv_layers)
            ]
        )
        for i in range(hidden_conv_layers):
            out_size = int((out_size - kernel_sizes[i] + 2 * paddings[i]) / strides[i]) + 1
        self.out_conv = nn.Conv2d(
            in_channels=n_filters[-1],
            out_channels=n_filters[-1],
            kernel_size=kernel_sizes[-1],
            padding=paddings[-1]
        )
        out_size = int((out_size - kernel_sizes[-1] + 2 * paddings[-1]) / strides[-1]) + 1
        self.n_flat = out_size ** 2 * n_filters[-1]
        self.linear = nn.Linear(in_features=self.n_flat, out_features=1)
        self.activation = act_func
        self.type(dtype)

    def forward(self, q, p):
        """Forward pass that returns the Hamiltonian for the given q and p inputs.

        q and p must be two (batch_size, channels, height, width) tensors.

        Args:
            q (torch.Tensor): The tensor corresponding to the position in abstract space.
            p (torch.Tensor): The tensor corresponding to the momentum in abstract space.

        Returns:
            A (batch_size, 1) shaped tensor with the Hamiltonian for each input in the batch.
        """
        x = torch.cat((q, p), dim=1)  # Concatenate q and p to obtain a N x 2C x H x W tensor
        x = self.activation(self.in_conv(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.out_conv(x))
        x = x.view(-1, self.n_flat)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    hamiltonian_net = HamiltonianNet(in_shape=(16, 4, 4))
    q, p = torch.randn((2, 128, 16, 4, 4))
    h = hamiltonian_net(q, p)
