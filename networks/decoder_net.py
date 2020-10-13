import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A residual block that up-samples the input image by a factor of 2.
    """

    def __init__(self, in_channels, n_filters=64, kernel_size=3, padding='same'):
        """Instantiate the residual block, composed by a 2x up-sampling and two convolutional
        layers.

        Args:
            in_channels (int): Number of input channels.
            n_filters (int): Number of filters, and thus output channels.
            kernel_size (int): Size of the convolutional kernels.
            padding (Union[str, int]): Padding for the convolutional kernels. Can be an int,
                or 'same' to automatically compute the padding to preserve the image size.
        """
        super().__init__()
        self.channels = in_channels
        self.n_filters = n_filters
        if isinstance(padding, int):
            use_padding = padding
        elif padding == 'same':
            use_padding = int(kernel_size / 2)
        else:
            raise ValueError('Padding argument not understood. Must be integer or "same".')
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=use_padding,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=use_padding,
            bias=False
        )
        if in_channels != n_filters:
            self.dim_match_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=1,
                bias=False,
                padding=0
            )
        self.leaky_relu = nn.LeakyReLU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply 2x up-sampling, followed by two convolutional layers with leaky relu. A sigmoid
        activation is applied at the end.

        Args:
            x (torch.Tensor): Input image of shape (N, C, H, W) where N is the batch size and C
                is the number of in_channels.

        Returns:
            A torch.Tensor with the upsampled images, of shape (N, n_filters, H, W).
        """
        x = self.upsample(x)
        residual = self.dim_match_conv(x) if self.channels != self.n_filters else x
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

    def __init__(self, in_channels, out_channels, n_filters):
        """Create the decoder network.

        Args:
            in_channels (int): Number of input encodings channels.
            out_channels (int): Number output image channels (1 for grayscale, 3 for RGB).
            n_filters (int): Number of filters in the intermediate convolutional layers.
        """
        super().__init__()
        self.res1 = ResidualBlock(in_channels=in_channels, n_filters=n_filters)
        self.res2 = ResidualBlock(in_channels=n_filters, n_filters=n_filters)
        self.res3 = ResidualBlock(in_channels=n_filters, n_filters=n_filters)
        self.out = nn.Conv2d(
            in_channels=n_filters,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply the three residual blocks and the final convolutional layer.

        Args:
            x (torch.Tensor): Tensor of shape (N, in_channels, H, W) where N is the batch size.

        Returns:
            Tensor of shape (out_channels, H * 2^3, W * 2^3) with the reconstructed image.
        """
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.sigmoid(self.out(x))
        return x


if __name__ == '__main__':
    decoder = DecoderNet(in_channels=16, out_channels=3, n_filters=64)
    inp = torch.randn((1, 16, 4, 4))
    out = decoder(inp)
    print(out.size())