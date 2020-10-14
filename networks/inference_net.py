"""This module contains the implementation of the inference step of the Hamiltonian Generative
Networks paper. The inference step is composed by an Encoder network and a Transformer network.
The encoder maps the input sequence of frames into a latent distribution and samples a Tensor z
from it using the reparametrization trick. The Transformer network takes the latent Tensor z and
maps it into the abstract phase space (q, p).
"""

import torch
import torch.nn as nn


class EncoderNet(nn.Module):
    """Implementation of the encoder network, that encodes the input frames sequence into a
    distribution over the latent space and samples with the common reparametrization trick.

    The network expects the RGB values to be concatenated. This means that if image shape is
    (3, 32, 32) and we have 10 images in a sequence, the network will accept an input of shape
    (N, 3 * 10, 32, 32), where N is the batch size.
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 6,
        'n_filters': [32, 64, 64, 64, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3],
        'strides': [1, 1, 1, 1, 1, 1, 1, 1],
    }

    def __init__(self, seq_len, in_channels, out_channels, hidden_conv_layers=None,
                 n_filters=None, kernel_sizes=None, strides=None, act_func=nn.ReLU(),
                 dtype=torch.float):
        """Instantiate the convolutional layers that compose the input network with the
        appropriate shapes.

        If K is the total number of layers, then hidden_conv_layers = K - 2. The length of
        n_filters must be K - 1, and that of kernel_sizes and strides must be K. If all
        them are None, EncoderNet.DEFAULT_PARAMS will be used.

        Args:
            seq_len (int): Number of frames that compose a sequence.
            in_channels (int): Number of channels of images in the input sequence.
            out_channels (int): Number of in_channels of the output latent encoding.
            hidden_conv_layers (int): Number of hidden convolutional layers (excluding the input
                and the two output layers for mean and variance).
            n_filters (list): List with number of filters for each of the hidden layers.
            kernel_sizes (list): List with kernel sizes for each convolutional layer.
            strides (list): List with strides for each convolutional layer.
            act_func (torch.nn.Module): The activation function to apply after each layer.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        if all(var is None for var in (hidden_conv_layers, n_filters, kernel_sizes, strides)):
            hidden_conv_layers = EncoderNet.DEFAULT_PARAMS['hidden_conv_layers']
            n_filters = EncoderNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = EncoderNet.DEFAULT_PARAMS['kernel_sizes']
            strides = EncoderNet.DEFAULT_PARAMS['strides']
        elif all(var is not None for var in
                 (hidden_conv_layers, n_filters, kernel_sizes, strides)):
            # If no Nones, check consistency
            assert len(n_filters) == hidden_conv_layers + 1,\
                'n_filters must be a list of length hidden_conv_layers + 1 ' \
                '(' + str(hidden_conv_layers + 1) + ' in this case).'
            assert len(kernel_sizes) == hidden_conv_layers + 2 and \
                   len(strides) == hidden_conv_layers + 2, \
                   'kernel_sizes and strides must be lists with values foreach layer in the ' \
                   'network (' + str(hidden_conv_layers + 2) + ' in this case).'
        else:
            raise ValueError('Args hidden_conv_layers, n_filters, kernel_sizes, and strides'
                             'can only be either all None, or all defined by the user.')
        paddings = [int(k/2) for k in kernel_sizes]
        self.input_conv = nn.Conv2d(
            in_channels=seq_len * in_channels,
            out_channels=n_filters[0],
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            stride=strides[0]
        )
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
        self.out_mean = nn.Conv2d(
            in_channels=n_filters[-1],
            out_channels=out_channels,
            kernel_size=kernel_sizes[-1],
            padding=paddings[-1],
            stride=strides[-1]
        )
        self.out_logvar = nn.Conv2d(
            in_channels=n_filters[-1],
            out_channels=out_channels,
            kernel_size=kernel_sizes[-1],
            padding=paddings[-1],
            stride=strides[-1]
        )
        self.activation = act_func
        self.type(dtype)

    def forward(self, x):
        """Compute the encoding of the given sequence of images.

        Args:
            x (torch.Tensor): A N x S x H x W tensor containing the sequence of frames. N is the
                batch size, S the number of frames in a sequence, H and W the height and width.

        Returns:
            A tuple (z, mu, stddev), which are all N x 48 x H x W tensors. z is the latent encoding
            for the given input sequence, while mu and stddev are distribution parameters.
        """
        x = self.activation(self.input_conv(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        mean = self.activation(self.out_mean(x))
        stddev = torch.exp(0.5 * self.activation(self.out_logvar(x)))
        epsilon = torch.randn_like(mean)
        z = mean + stddev * epsilon
        return z, mean, stddev


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

    def __init__(self, in_channels, out_channels, hidden_conv_layers=None,
                 n_filters=None, kernel_sizes=None, strides=None, act_func=torch.nn.ReLU(),
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
        if all(var is None for var in (hidden_conv_layers, n_filters, kernel_sizes, strides)):
            hidden_conv_layers = TransformerNet.DEFAULT_PARAMS['hidden_conv_layers']
            n_filters = TransformerNet.DEFAULT_PARAMS['n_filters']
            kernel_sizes = TransformerNet.DEFAULT_PARAMS['kernel_sizes']
            strides = TransformerNet.DEFAULT_PARAMS['strides']
        elif all(var is not None for var in (hidden_conv_layers, n_filters, kernel_sizes, strides)):
            # If no Nones, check consistency
            assert len(n_filters) == hidden_conv_layers + 1,\
                'n_filters must be of length hidden_conv_layers + 1 ' \
                '(' + str(hidden_conv_layers + 1) + ' in this case).'
            assert len(kernel_sizes) == hidden_conv_layers + 2 \
                   and len(strides) == hidden_conv_layers + 2, \
                   'kernel_sizes and strides must be lists with values foreach layer in the ' \
                   'network (' + str(hidden_conv_layers + 2) + ' in this case).'
        else:
            raise ValueError('Args hidden_conv_layers, n_filters, kernel_sizes, and strides'
                             'can only be either all None, or all defined by the user.')

        paddings = [int(k/2) for k in kernel_sizes]
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters[0],
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            stride=strides[0]
        )
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
        self.out_conv = nn.Conv2d(
            in_channels=n_filters[-1],
            out_channels=out_channels * 2,
            kernel_size=kernel_sizes[-1],
            padding=paddings[-1],
            stride=strides[-1]
        )
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
        q, p = to_phase_space(x)
        return q, p


def to_phase_space(encoding):
    """Takes the encoder-transformer output and returns the q and p tensors.

    Args:
        encoding (torch.Tensor): A tensor of shape (batch_size, channels, ...).

    Returns:
        Two tensors of shape (batch_size, channels/2, ...) resulting from splitting the given
        tensor along the second dimension.
    """
    assert encoding.shape[1] % 2 == 0, 'The number of in_channels is odd. Cannot split properly.'
    half_len = int(encoding.shape[1] / 2)
    q = encoding[:, :half_len]
    p = encoding[:, half_len:]
    return q, p


def concat_rgb(sequences_batch):
    """Concatenate the images along channel dimension.

    Args:
        sequences_batch (torch.Tensor): A Tensor with shape (batch_size, seq_len, channels, height, width)
            containing the images of the sequence.

    Returns:
        A Tensor with shape (batch_size, seq_len * channels, height, width) with the images
        concatenated along the channel dimension.
    """
    batch_size, seq_len, channels, h, w = sequences_batch.size()
    return torch.reshape(sequences_batch, shape=(batch_size, seq_len * channels, h, w))


if __name__ == '__main__':
    encoder = EncoderNet(seq_len=10, in_channels=3, out_channels=48, hidden_conv_layers=9,
                         kernel_sizes=[3 for i in range(11)],
                         n_filters=[64 for i in range(10)],
                         strides=[1 for i in range(11)])
    transformer = TransformerNet(in_channels=48, out_channels=32, hidden_conv_layers=4,
                                 kernel_sizes=[3, 3, 3, 3, 3, 4],
                                 n_filters=[32, 48, 64, 96, 128],
                                 strides=[2, 2, 2, 1, 1, 2])

    inp = torch.randn((128, 10, 3, 32, 32))
    inp = concat_rgb(inp)
    encoded, mean, var = encoder(inp)

    q, p = transformer(encoded)
    print(q.size())
    print(p.size())

