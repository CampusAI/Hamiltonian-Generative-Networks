"""This module contains the implementation of the Encoder step of the Hamiltonian Generative Networks
paper. The encoder maps the input sequence of frames into a latent distribution and samples a Tensor z
from it using the re-parametrization trick. 
"""

import torch
import torch.nn as nn


class EncoderNet(nn.Module):
    """Implementation of the encoder network, that encodes the input frames sequence into a
    distribution over the latent space and samples with the common reparametrization trick.

    The network expects the images to be concatenated along channel dimension. This means that if
    a batch of sequences has shape (batch_size, seq_len, channels, height, width) the network
    will accept an input of shape (batch_size, seq_len * channels, height, width).
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 6,
        'n_filters': [32, 64, 64, 64, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3],
        'strides': [1, 1, 1, 1, 1, 1, 1, 1],
    }

    def __init__(self,
                 seq_len,
                 in_channels,
                 out_channels,
                 hidden_conv_layers=None,
                 n_filters=None,
                 kernel_sizes=None,
                 strides=None,
                 act_func=nn.ReLU(),
                 dtype=torch.float):
        """Instantiate the convolutional layers that compose the input network with the
        appropriate shapes.

        If K is the total number of layers, then hidden_conv_layers = K - 2. The length of
        n_filters must be K - 1, and that of kernel_sizes and strides must be K. If all
        them are None, EncoderNet.DEFAULT_PARAMS will be used.

        Args:
            seq_len (int): Number of frames that compose a sequence.
            in_channels (int): Number of channels of images in the input sequence.
            out_channels (int): Number of channels of the output latent encoding.
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
            hidden_conv_layers = EncoderNet.DEFAULT_PARAMS[
                'hidden_conv_layers']
            n_filters = EncoderNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = EncoderNet.DEFAULT_PARAMS['kernel_sizes']
            strides = EncoderNet.DEFAULT_PARAMS['strides']
        elif all(var is not None for var in (hidden_conv_layers, n_filters,
                                             kernel_sizes, strides)):
            # If no Nones, check consistency
            assert len(n_filters) == hidden_conv_layers + 1,\
                'n_filters must be a list of length hidden_conv_layers + 1 ' \
                '(' + str(hidden_conv_layers + 1) + ' in this case).'
            assert len(kernel_sizes) == hidden_conv_layers + 2 and \
                   len(strides) == hidden_conv_layers + 2, \
                   'kernel_sizes and strides must be lists with values foreach layer in the ' \
                   'network (' + str(hidden_conv_layers + 2) + ' in this case).'
        else:
            raise ValueError(
                'Args hidden_conv_layers, n_filters, kernel_sizes, and strides'
                'can only be either all None, or all defined by the user.')
        paddings = [int(k / 2) for k in kernel_sizes]
        self.input_conv = nn.Conv2d(in_channels=seq_len * in_channels,
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
        self.out_mean = nn.Conv2d(in_channels=n_filters[-1],
                                  out_channels=out_channels,
                                  kernel_size=kernel_sizes[-1],
                                  padding=paddings[-1],
                                  stride=strides[-1])
        self.out_logvar = nn.Conv2d(in_channels=n_filters[-1],
                                    out_channels=out_channels,
                                    kernel_size=kernel_sizes[-1],
                                    padding=paddings[-1],
                                    stride=strides[-1])
        self.activation = act_func
        self.type(dtype)

    def forward(self, x, sample=True):
        """Compute the encoding of the given sequence of images.

        Args:
            x (torch.Tensor): A (batch_size, seq_len * channels, height, width) tensor containing
            the sequence of frames.
            sample (bool): Whether to sample from the encoding distribution or returning the mean.

        Returns:
            A tuple (z, mu, log_var), which are all N x 48 x H x W tensors. z is the latent encoding
            for the given input sequence, while mu and log_var are distribution parameters.
        """
        x = self.activation(self.input_conv(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        mean = self.out_mean(x)
        if not sample:
            return mean, None, None  # Return None to ensure that they're not used in loss
        log_var = self.out_logvar(x)
        stddev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        z = mean + stddev * epsilon
        return z, mean, log_var