import pytest
import torch

from networks import decoder_net


def test_decoder_shape_default_params():
    batch_size = 10
    # With default params
    decoder = decoder_net.DecoderNet(in_channels=16, out_channels=3)
    inp = torch.randn((batch_size, 16, 4, 4))
    out = decoder(inp)
    expected_size = torch.Size([batch_size, 3, 32, 32])
    actual_size = out.size()
    assert expected_size == actual_size

    # With default params, different in-out channels
    decoder = decoder_net.DecoderNet(in_channels=32, out_channels=10)
    inp = torch.randn((batch_size, 32, 4, 4))
    out = decoder(inp)
    expected_size = torch.Size([batch_size, 10, 32, 32])
    actual_size = out.size()
    assert expected_size == actual_size

    # With custom params
    decoder = decoder_net.DecoderNet(in_channels=16, out_channels=3, n_residual_blocks=4,
                                     n_filters=[32, 64, 128, 256],
                                     kernel_sizes=[3, 3, 3, 5, 5])
    inp = torch.randn((batch_size, 16, 4, 4))
    out = decoder(inp)
    expected_size = torch.Size([batch_size, 3, 64, 64])
    actual_size = out.size()
    assert expected_size == actual_size

    # With custom params, different in-out shape
    decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=4,
                                     n_filters=[32, 64, 128, 256],
                                     kernel_sizes=[3, 3, 3, 5, 5])
    inp = torch.randn((batch_size, 8, 4, 4))
    out = decoder(inp)
    expected_size = torch.Size([batch_size, 1, 64, 64])
    actual_size = out.size()
    assert expected_size == actual_size


def test_decoder_raises_exception():
    """Test the encoder correctly raises exceptions for wrong custom params
    """
    with pytest.raises(ValueError):
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=4,
                                         n_filters=[32, 64, 128, 256])
    with pytest.raises(ValueError):
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=4,
                                         kernel_sizes=[3, 3, 3, 5, 5])
    with pytest.raises(ValueError):
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1,
                                         n_filters=[32, 64, 128, 256],
                                         kernel_sizes=[3, 3, 3, 5, 5])
    with pytest.raises(AssertionError):  # Wrong n_filters and kernel_sizes
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=3,
                                         n_filters=[32, 64, 128, 256],
                                         kernel_sizes=[3, 3, 3, 5, 5])
    with pytest.raises(AssertionError):  # Wrong n_filters
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=4,
                                         n_filters=[32, 64, 128],
                                         kernel_sizes=[3, 3, 3, 5, 5])
    with pytest.raises(AssertionError):  # Wrong kernel_sizes
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=4,
                                         n_filters=[32, 64, 128, 256],
                                         kernel_sizes=[3, 3, 3])
    with pytest.raises(AssertionError):  # Wrong n_filters for no residual blocks
        decoder = decoder_net.DecoderNet(in_channels=8, out_channels=1, n_residual_blocks=0,
                                         n_filters=[3],
                                         kernel_sizes=[5])