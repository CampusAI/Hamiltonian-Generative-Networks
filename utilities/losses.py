import torch

def reconstruction_loss(target, prediction):
    """Computes the MSE loss between the target and the predictions.
        
    Args:
        target (Tensor): The target batch
        prediction (Tensor) The prediction of the model

    Returns:
        (Tensor): MSE loss
    """
    # Compute the sum error across each channel in the batch and then average
    # last_dim = len(target.shape) - 1
    # loss_per_channel = torch.sum(torch.pow(prediction - target, 2), [last_dim - 1, last_dim])
    # return torch.mean(loss_per_channel)
    return torch.mean(torch.pow(prediction - target, 2))

def kld_loss(mu, logvar):
    """ First it computes the KLD over each datapoint in the batch as a sum over all latent dims. 
        It returns the mean KLD over the batch size.
        The KLD is computed in comparison to a multivariate Gaussian with zero mean and identity covariance.

    Args:
        mu (torch.Tensor): the part of the latent vector that corresponds to the mean
        logvar (torch.Tensor): the log of the variance (sigma squared)

    Returns:
        (torch.Tensor): KL divergence.
    """
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))

def geco_constraint(target, prediction, tol):
    """Computes the constraint for the geco algorithm
    """
    return reconstruction_loss(prediction, target) - tol**2