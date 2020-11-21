import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.losses import kld_loss

def test_kld_loss():
    batch_sizes = [1, 10, 100]
    latent_size = 8
    for batch_size in [10]: #batch_sizes:
        mu = torch.randn((batch_size, latent_size))
        logvar = torch.randn((batch_size, latent_size))

        kld = kld_loss(mu, logvar)
        assert kld.dim() == 0

        mu = torch.randn((batch_size, latent_size, latent_size, latent_size))
        logvar = torch.randn((batch_size, latent_size, latent_size, latent_size))

        kld = kld_loss(mu, logvar)
        assert kld.dim() == 0

test_kld_loss()