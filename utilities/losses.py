import torch


def reconstruction_loss(prediction, target):
    """Computes the MSE loss between the target and the predictions.
        
    Args:
        prediction (Tensor) The prediction of the model
        target (Tensor): The target batch

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
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


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
