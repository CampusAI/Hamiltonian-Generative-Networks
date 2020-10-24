import torch


def reconstruction_loss(target, prediction):
    """Computes the MSE loss between the target and the predictions.
        
    Args:
        target (Tensor): The target batch
        prediction (Tensor) The prediction of the model

    Returns:
        (Tensor): MSE loss
    """
    mse = torch.nn.MSELoss()
    return mse(input=prediction, target=target)


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
    return torch.mean(-0.5 *
                      torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))


def geco_constraint(target, prediction, tol):
    """Computes the constraint for the geco algorithm
    """
    rec_loss = reconstruction_loss(prediction, target)
    return rec_loss - tol**2, rec_loss