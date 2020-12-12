import torch


def reconstruction_loss(prediction, target, mean_reduction=True):
    """Computes the MSE loss between the target and the predictions.
        
    Args:
        prediction (Tensor) The prediction of the model
        target (Tensor): The target batch
        mean_reduction (bool): Whether to perform mean reduction across batch (default is true)
    Returns:
        (Tensor): MSE loss
    """
    reduction = 'mean' if mean_reduction else 'none'
    mse = torch.nn.MSELoss(reduction=reduction)
    if mean_reduction:
        return mse(input=prediction, target=target)
    else:
        return mse(input=prediction, target=target).flatten(1).mean(-1)


def kld_loss(mu, logvar, mean_reduction=True):
    """ First it computes the KLD over each datapoint in the batch as a sum over all latent dims. 
        It returns the mean KLD over the batch size.
        The KLD is computed in comparison to a multivariate Gaussian with zero mean and identity covariance.

    Args:
        mu (torch.Tensor): the part of the latent vector that corresponds to the mean
        logvar (torch.Tensor): the log of the variance (sigma squared)
        mean_reduction (bool): Whether to perform mean reduction across batch (default is true)

    Returns:
        (torch.Tensor): KL divergence.
    """
    mu = mu.flatten(1)
    logvar = logvar.flatten(1)
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    if mean_reduction:
        return torch.mean(kld_per_sample, dim = 0)
    else:
        return kld_per_sample


def geco_constraint(target, prediction, tol):
    """Computes the constraint for the geco algorithm.

    Args:
        target (toch.Tensor): the rollout target.
        prediction (torch.Tensor): the prediction of the model.
        tol (float): the tolerance we accept between target and prediction.

    Returns:
        (tuple(torch.Tensor, torch.Tensor)): the constraing value as MSE minus the tolerance, and MSE.
    """
    rec_loss = reconstruction_loss(prediction=prediction, target=target)
    return rec_loss - tol**2, rec_loss
