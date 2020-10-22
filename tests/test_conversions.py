import torch
import numpy as np

from utilities import conversions


def test_to_channels_last():
    batch_size, seq_len, height, width, channels = 5, 5, 32, 32, 3
    tensor = torch.randn((batch_size, seq_len, channels, height, width))
    converted = conversions.to_channels_last(tensor)

    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        assert tensor[b, s, c, h, w] == converted[b, s, h, w, c]


def test_to_channels_first():
    batch_size, seq_len, height, width, channels = 5, 5, 32, 32, 3
    tensor = torch.randn((batch_size, seq_len, height, width, channels))
    converted = conversions.to_channels_first(tensor)

    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        assert tensor[b, s, h, w, c] == converted[b, s, c, h, w]


def test_concat_rgb():
    """Test that concat_rgb correctly reshapes the tensor by concatenating sequences along
    channel dimensions
    """
    batch_len = 2
    seq_len = 5
    channels = 3
    img_size = 32
    batch = torch.randn((batch_len, seq_len, channels, img_size, img_size))
    concatenated = conversions.concat_rgb(batch)

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


def test_batch_to_sequence():
    batch_size, seq_len, height, width, channels = 15, 10, 32, 32, 3
    batch = np.random.normal(size=(batch_size, seq_len, height, width,
                                   channels))
    sequence = conversions.batch_to_sequence(batch)

    for b in range(batch_size):
        for s in range(seq_len):
            assert np.array_equal(batch[b],
                                  sequence[b * seq_len:(b + 1) * seq_len])
