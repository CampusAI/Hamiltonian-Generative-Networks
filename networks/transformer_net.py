"""This module contains the implementation of the Transformer step of the Hamiltonian Generative Networks paper.
The Transformer network takes the latent Tensor z and maps it into the abstract phase space (q, p).
"""

import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    """Implementation of the encoder-transformer network, that maps the latent space into the
    phase space.
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 2,
        'n_filters': [64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [2, 2, 2, 1],
    }

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_conv_layers=None,
                 n_filters=None,
                 kernel_sizes=None,
                 strides=None,
                 act_func=torch.nn.ReLU(),
                 dtype=torch.float):
        """Instantiate the convolutional layers with the given attributes or using the default
        parameters.

        If K is the total number of layers, then hidden_conv_layers = K - 2. The length of
        n_filters must be K - 1, and that of kernel_sizes and strides must be K. If all
        them are None, TransformerNet.DEFAULT_PARAMS will be used.


        Args:
            in_channels (int): Number of input in_channels.
            out_channels (int): Number of in_channels of q and p
            hidden_conv_layers (int): Number of hidden convolutional layers (excluding the input
                and the two output layers for mean and variance).
            n_filters (list): List with number of filters for each of the hidden layers.
            kernel_sizes (list): List with kernel sizes for each convolutional layer.
            strides (list): List with strides for each convolutional layer.
            act_func (torch.nn.Module): The activation function to apply after each layer.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        if all(var is None for var in (hidden_conv_layers, n_filters,
                                       kernel_sizes, strides)):
            hidden_conv_layers = TransformerNet.DEFAULT_PARAMS[
                'hidden_conv_layers']
            n_filters = TransformerNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = TransformerNet.DEFAULT_PARAMS['kernel_sizes']
            strides = TransformerNet.DEFAULT_PARAMS['strides']
        elif all(var is not None for var in (hidden_conv_layers, n_filters,
                                             kernel_sizes, strides)):
            # If no Nones, check consistency
            assert len(n_filters) == hidden_conv_layers + 1,\
                'n_filters must be of length hidden_conv_layers + 1 ' \
                '(' + str(hidden_conv_layers + 1) + ' in this case).'
            assert len(kernel_sizes) == hidden_conv_layers + 2 \
                   and len(strides) == hidden_conv_layers + 2, \
                   'kernel_sizes and strides must be lists with values foreach layer in the ' \
                   'network (' + str(hidden_conv_layers + 2) + ' in this case).'
        else:
            raise ValueError(
                'Args hidden_conv_layers, n_filters, kernel_sizes, and strides'
                'can only be either all None, or all defined by the user.')

        paddings = [int(k / 2) for k in kernel_sizes]
        self.in_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=n_filters[0],
                                 kernel_size=kernel_sizes[0],
                                 padding=paddings[0],
                                 stride=strides[0])
        self.hidden_layers = nn.ModuleList(modules=[
            nn.Conv2d(in_channels=n_filters[i],
                      out_channels=n_filters[i + 1],
                      kernel_size=kernel_sizes[i + 1],
                      padding=paddings[i + 1],
                      stride=strides[i + 1]) for i in range(hidden_conv_layers)
        ])
        self.out_conv = nn.Conv2d(in_channels=n_filters[-1],
                                  out_channels=out_channels * 2,
                                  kernel_size=kernel_sizes[-1],
                                  padding=paddings[-1],
                                  stride=strides[-1])
        self.activation = act_func
        self.type(dtype)

    def forward(self, x):
        """Transforms the given encoding into two tensors q, p.

        Args:
            x (torch.Tensor): A Tensor of shape (batch_size, channels, H, W).

        Returns:
            Two Tensors q, p corresponding to vectors of abstract positions and momenta.
        """
        x = self.activation(self.in_conv(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.out_conv(x))
        q, p = self.to_phase_space(x)
        return q, p

    @staticmethod
    def to_phase_space(encoding):
        """Takes the encoder-transformer output and returns the q and p tensors.

        Args:
            encoding (torch.Tensor): A tensor of shape (batch_size, channels, ...).

        Returns:
            Two tensors of shape (batch_size, channels/2, ...) resulting from splitting the given
            tensor along the second dimension.
        """
        assert encoding.shape[1] % 2 == 0,\
            'The number of in_channels is odd. Cannot split properly.'
        half_len = int(encoding.shape[1] / 2)
        q = encoding[:, :half_len]
        p = encoding[:, half_len:]
        return q, p
