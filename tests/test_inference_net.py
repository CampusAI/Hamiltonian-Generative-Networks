"""This module provides tests for the network architectures. Requires pytest (pip install pytest).

Run by command line:
    pytest tests/networks.py
"""

import torch
import pytest

from networks import inference_net
from networks import hamiltonian_net
from networks import decoder_net


def test_concat_rgb():
    """Test that concat_rgb correctly reshapes the tensor by concatenating sequences along
    channel dimensions
    """
    batch_len = 2
    seq_len = 5
    channels = 3
    img_size = 32
    batch = torch.randn((batch_len, seq_len, channels, img_size, img_size))
    concatenated = inference_net.concat_rgb(batch)

    for concat_seq, batch_seq in zip(concatenated, batch):
        expected = torch.empty((seq_len * channels, 32, 32))
        for i in range(32):
            for j in range(32):
                rgb = torch.empty(channels * seq_len)
                for s in range(seq_len):
                    rgb[s * channels + 0] = batch_seq[s, 0, i, j]
                    rgb[s * channels + 1] = batch_seq[s, 1, i, j]
                    rgb[s * channels + 2] = batch_seq[s, 2, i, j]
                expected[:, i, j] = rgb
        assert torch.equal(expected, concat_seq)


def test_to_phase_space():
    batch_size = 10
    channels = 48
    img_size = 32
    batch = torch.randn((batch_size, channels, img_size, img_size))
    q, p = inference_net.to_phase_space(batch)

    for k in range(batch_size):
        for i in range(int(channels / 2)):
            for h in range(img_size):
                for w in range(img_size):
                    assert q[k, i, h, w] == batch[k, i, h, w]
                    assert p[k, i, h, w] == batch[k, int(channels / 2) + i, h, w]


def test_encoder_out_shape():
    seq_len = 10
    img_size = 32
    in_channels = 3
    out_channels = 48
    hidden_conv_layers = 4
    n_filters = [32, 48, 64, 80, 96]  # Must be hidden_conv_layers + 1
    kernel_sizes = [3, 5, 7, 7, 5, 3]  # Must be hidden_conv_layers + 2
    strides = [1, 2, 1, 2, 1, 1]  # Must be hidden_conv_layers + 2
    encoder = inference_net.EncoderNet(
        seq_len=seq_len,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_conv_layers=hidden_conv_layers,
        n_filters=n_filters,
        kernel_sizes=kernel_sizes,
        strides=strides
    )

    expected_out_size = torch.Size([128, 48, int(img_size/4), int(img_size/4)])

    inputs = torch.randn((128, seq_len, in_channels, img_size, img_size))
    inputs = inference_net.concat_rgb(inputs)
    z, mu, var = encoder(inputs)

    assert z.size() == expected_out_size


def test_encoder_raises_exception():
    """Test that the encoder correctly raises exceptions if called with wrong params.
    """
    seq_len = 10
    img_size = 32
    in_channels = 3
    out_channels = 48
    hidden_conv_layers = 4
    # Args to be tested
    n_filters = [32, 48, 64, 80, 96]  # Must be hidden_conv_layers + 1
    kernel_sizes = [3, 5, 7, 7, 5, 3]  # Must be hidden_conv_layers + 2
    strides = [1, 2, 1, 2, 1, 1]  # Must be hidden_conv_layers + 2
    with pytest.raises(AssertionError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters[:-1],  # n_filters is shorter than it should be
            kernel_sizes=kernel_sizes,
            strides=strides
        )
    with pytest.raises(AssertionError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes[:-1],  # kernel_sizes is shorter than it should be
            strides=strides
        )
    with pytest.raises(AssertionError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            strides=strides[:-1]  # strides is shorter than it should be
        )
    with pytest.raises(AssertionError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters+[64],  # n_filters is longer than it should be
            kernel_sizes=kernel_sizes,
            strides=strides
        )
    with pytest.raises(AssertionError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes+[5],  # kernel_sizes is longer than it should be
            strides=strides
        )
    with pytest.raises(AssertionError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            strides=strides+[2]  # strides is longer than it should be
        )
    # Test not all arguments are given
    with pytest.raises(ValueError):
        encoder = inference_net.EncoderNet(
            seq_len=seq_len,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            # Missing strides
        )
    # Test that correctly works if no args are given
    encoder = inference_net.EncoderNet(
        seq_len=seq_len,
        in_channels=in_channels,
        out_channels=out_channels
    )


def test_transformer_shape():
    img_size = 32
    in_channels = 48
    out_channels = 16  # For q and p separately
    hidden_conv_layers = 4
    n_filters = [32, 48, 64, 80, 96]  # Must be hidden_conv_layers + 1
    kernel_sizes = [3, 5, 7, 7, 5, 3]  # Must be hidden_conv_layers + 2
    strides = [1, 1, 1, 1, 1, 1]  # Must be hidden_conv_layers + 2
    transformer = inference_net.TransformerNet(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_conv_layers=hidden_conv_layers,
        n_filters=n_filters,
        kernel_sizes=kernel_sizes,
        strides=strides
    )

    expected_out_size = torch.Size([128, 16, img_size, img_size])

    inputs = torch.randn((128, in_channels, img_size, img_size))
    q, p = transformer(inputs)

    assert q.size() == expected_out_size
    assert p.size() == expected_out_size


def test_transformer_raises_exception():
    """Test that the transformer correctly raises exceptions if called with wrong params.
    """
    img_size = 32
    in_channels = 3
    out_channels = 48
    hidden_conv_layers = 4
    # Args to be tested
    n_filters = [32, 48, 64, 80, 96]  # Must be hidden_conv_layers + 1
    kernel_sizes = [3, 5, 7, 7, 5, 3]  # Must be hidden_conv_layers + 2
    strides = [1, 2, 1, 2, 1, 1]  # Must be hidden_conv_layers + 2
    with pytest.raises(AssertionError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters[:-1],  # n_filters is shorter than it should be
            kernel_sizes=kernel_sizes,
            strides=strides
        )
    with pytest.raises(AssertionError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes[:-1],  # kernel_sizes is shorter than it should be
            strides=strides
        )
    with pytest.raises(AssertionError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            strides=strides[:-1]  # strides is shorter than it should be
        )
    with pytest.raises(AssertionError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters + [64],  # n_filters is longer than it should be
            kernel_sizes=kernel_sizes,
            strides=strides
        )
    with pytest.raises(AssertionError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes + [5],  # kernel_sizes is longer than it should be
            strides=strides
        )
    with pytest.raises(AssertionError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            strides=strides + [2]  # strides is longer than it should be
        )
    # Test not all arguments are given
    with pytest.raises(ValueError):
        transformer = inference_net.TransformerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_conv_layers=hidden_conv_layers,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            # Missing strides
        )
    # Test that correctly works if no args are given
    transformer = inference_net.TransformerNet(
        in_channels=in_channels,
        out_channels=out_channels
    )
