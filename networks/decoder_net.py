"""This module contains the implementation of a decoder network, that applies 3 residual blocks
to the input abstract position q. In the paper q is a (16, 4, 4) tensor that can be seen as a 4x4
image with 16 channels, but here other sizes may be used.
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A residual block that up-samples the input image by a factor of 2.
    """
    def __init__(self, in_channels, n_filters=64, kernel_size=3, dtype=torch.float):
        """Instantiate the residual block, composed by a 2x up-sampling and two convolutional
        layers.

        Args:
            in_channels (int): Number of input channels.
            n_filters (int): Number of filters, and thus output channels.
            kernel_size (int): Size of the convolutional kernels.
            dtype (torch.dtype): Type to be used in tensors.
        """
        super().__init__()
        self.channels = in_channels
        self.n_filters = n_filters
        padding = int(kernel_size / 2)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        if in_channels != n_filters:
            self.dim_match_conv = nn.Conv2d(in_channels=in_channels,
                                            out_channels=n_filters,
                                            kernel_size=1,
                                            padding=0)
        self.leaky_relu = nn.LeakyReLU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()
        self.type(dtype)

    def forward(self, x):
        """Apply 2x up-sampling, followed by two convolutional layers with leaky relu. A sigmoid
        activation is applied at the end.

        TODO: Should we use batch normalization? It is often common in residual blocks.
        TODO: Here we apply a convolutional layer to the input up-sampled tensor if its number
            of channels does not match the convolutional layer channels. Is this the correct way?

        Args:
            x (torch.Tensor): Input image of shape (N, C, H, W) where N is the batch size and C
                is the number of in_channels.

        Returns:
            A torch.Tensor with the up-sampled images, of shape (N, n_filters, H, W).
        """
        x = self.upsample(x)
        residual = self.dim_match_conv(
            x) if self.channels != self.n_filters else x
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.sigmoid(x + residual)
        return x


class DecoderNet(nn.Module):
    """The Decoder network, that takes a latent encoding of shape (in_channels, H, W)
    and produces the output image by applying 3 ResidualBlock modules and a final 1x1 convolution.
    Each residual block up-scales the image by 2, and the convolution produces the desired number
    of output channels, thus the output shape is (out_channels, H*2^3, W*2^3).
    """

    DEFAULT_PARAMS = {
        'n_residual_blocks': 3,
        'n_filters': [64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3],
    }

    def __init__(self,
                 in_channels,
                 out_channels=3,
                 n_residual_blocks=None,
                 n_filters=None,
                 kernel_sizes=None,
                 dtype=torch.float):
        """Create the decoder network composed of the given number of residual blocks.

        Args:
            in_channels (int): Number of input encodings channels.
            out_channels (int): Number output image channels (1 for grayscale, 3 for RGB).
            n_residual_blocks (int): Number of residual blocks in the network.
            n_filters (list): List where the i-th element is the number of filters for
                convolutional layers for the i-th residual block, excluding the output block.
                Therefore, n_filters must be of length n_residual_blocks - 1
            kernel_sizes(list): List where the i-th element is the kernel size of convolutional
                layers for the i-th residual block.
        """
        super().__init__()
        if all(var is None for var in (n_residual_blocks, n_filters, kernel_sizes)):
            n_residual_blocks = DecoderNet.DEFAULT_PARAMS['n_residual_blocks']
            n_filters = DecoderNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = DecoderNet.DEFAULT_PARAMS['kernel_sizes']
        elif all(var is not None for var in (n_residual_blocks, n_filters, kernel_sizes)):
            assert len(kernel_sizes) == n_residual_blocks + 1, \
                'kernel_sizes and upsample must be of length n_residual_blocks + 1 ('\
                + str(n_residual_blocks + 1) + ' in this case).'
            assert len(n_filters) == n_residual_blocks, 'n_filters must be of length ' \
                'n_residual_blocks (' + str(n_residual_blocks) + ' in this case).'
        else:
            raise ValueError(
                'Args n_residual_blocks, n_filters, kernel_size, upsample '
                'can only be either all None, or all defined by the user.')
        filters = [in_channels] + n_filters
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=int(filters[i]),
                n_filters=int(filters[i + 1]),
                kernel_size=int(kernel_sizes[i]),
                dtype=dtype
            ) for i in range(n_residual_blocks)
        ])
        self.out_conv = nn.Conv2d(
            in_channels=filters[-1],
            out_channels=out_channels,
            kernel_size=kernel_sizes[-1],
            padding=int(kernel_sizes[-1] / 2)  # To not resize the image
        )
        self.sigmoid = nn.Sigmoid()
        self.type(dtype)

    def forward(self, x):
        """Apply the three residual blocks and the final convolutional layer.

        Args:
            x (torch.Tensor): Tensor of shape (N, in_channels, H, W) where N is the batch size.

        Returns:
            Tensor of shape (out_channels, H * 2^3, W * 2^3) with the reconstructed image.
        """
        for layer in self.residual_blocks:
            x = layer(x)
        x = self.sigmoid(self.out_conv(x))
        return x
