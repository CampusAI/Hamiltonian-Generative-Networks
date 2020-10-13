import torch
import torch.nn as nn


class EncoderNet(nn.Module):
    """Implementation of the encoder network, that encodes the input grayscale frames sequence
    into a latent space with the common variational reparametrization and sampling technique.
    """

    def __init__(self, seq_len, out_channels, dtype=torch.float):
        """Instantiate the 8 convolutional layers.

        Args:
            seq_len (int): Number of frames that compose a sequence.
            out_channels (int): Number of in_channels of the latent encoding.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels=seq_len,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.out_mean = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        self.out_logvar = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.type(dtype)

    def forward(self, x):
        """Compute the encoding of the given sequence of images.

        Args:
            x (torch.Tensor): A N x S x H x W tensor containing the sequence of frames. N is the
                batch size, S the number of frames in a sequence, H and W the height and width.

        Returns:
            A tuple z, mu, stddev, that are all N x 48 x H x W tensors. z is the latent encoding
            for the given input sequence, while mu and stddev are the parameters of the
            variational distribution.
        """
        x = self.input_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mean = self.out_mean(x)
        stddev = torch.exp(0.5 * self.out_logvar(x))
        epsilon = torch.randn_like(mean)
        z = mean + stddev * epsilon
        return z, mean, stddev


class TransformerNet(nn.Module):
    """Implementation of the encoder-transformer network, that maps the latent space into the
    phase space.
    """

    def __init__(self, in_channels, out_channels, dtype=torch.float):
        """Instantiate the 4 convolutional layers.

        Args:
            in_channels (int): Number of input in_channels.
            out_channels (int): Number of in_channels of the output.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=2
        )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.out_conv = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.type(dtype)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out_conv(x)
        q, p = to_phase_space(x)
        return q, p


def to_phase_space(encoding):
    """Takes the encoder-transformer output and returns the q and p tensors.

    Args:
        encoding (torch.Tensor): A (N, C, ...) tensor, where N is the batch size, C the number
            of in_channels.

    Returns:
        q and p, that are (N, C/2, ...) tensors.
    """
    assert encoding.shape[1] % 2 == 0, 'The number of in_channels is odd. Cannot split into q and p.'
    half_len = int(encoding.shape[1] / 2)
    q = encoding[:, :half_len]
    p = encoding[:, half_len:]
    return q, p
