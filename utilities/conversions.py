import torch
import numpy as np


def to_channels_last(tensor):
    """Convert a tensor from shape (batch_size, seq_len, channels, height, width) to
    shape (batch_size, seq_len, height, width, channels).

    Args:
        tensor (torch.Tensor): Tensor to be converted.

    Returns:
        A view of the given tensor with shape (batch_size, seq_len, height, channels, width).
        Any change applied to this out tensor will be applied to the input tensor.
    """
    return tensor.permute(0, 1, 3, 4, 2)


def to_channels_first(tensor):
    """Convert a tensor from shape (batch_size, seq_len, height, width, channels) to
    shape (batch_size, seq_len, channels, height, width).

    Args:
        tensor (torch.Tensor): Tensor to be converted.

    Returns:
        A view of the given tensor with shape (batch_size, seq_len, channels, height, width).
        Any change applied to this out tensor will be applied to the input tensor.
    """
    return tensor.permute(0, 1, 4, 2, 3)


def concat_rgb(batch):
    """Concatenate the images along channel dimension.

    Args:
        batch (torch.Tensor): A Tensor with shape (batch_size, seq_len, channels, height, width)
            containing the images of the sequence.

    Returns:
        A Tensor with shape (batch_size, seq_len * channels, height, width) with the images
        concatenated along the channel dimension.
    """
    batch_size, seq_len, channels, h, w = batch.size()
    return batch.reshape((batch_size, seq_len * channels, h, w))


def batch_to_sequence(batch):
    """Convert a batch of sequence of images into a single sequence composed by the concatenation
    of sequences in the batch.

    Args:
        batch (numpy.ndarray): Numpy array of sequences of images, must have shape
            (batch_size, seq_len, height, width, channels).

    Returns:
        A numpy array of shape (batch_size * seq_len, height, width, channels) with the
        concatenation of the given batch of sequences.
    """
    return np.concatenate(batch, axis=0)
