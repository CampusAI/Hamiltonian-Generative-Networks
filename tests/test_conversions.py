import torch

from utilities import conversions


def test_conversion():
    batch_size, seq_len, height, width, channels = 5, 5, 32, 32, 3
    tensor = torch.randn((batch_size, seq_len, channels, height, width))
    converted = conversions.to_channels_last(tensor)

    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        assert tensor[b, s, c, h, w] == converted[b, s, h, w, c]

