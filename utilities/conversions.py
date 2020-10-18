import torch


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
