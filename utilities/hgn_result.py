import torch

class HgnResult():
    """Class to bundle HGN guessed output information
    """
    def __init__(self):
        self.input = None
        self.z_mean = None
        self.z_std = None
        self.z_sample = None
        self.q_s = []
        self.p_s = []
        self.reconstructed_rollout = None

    def set_input(self, rollout):
        """Store ground truth of system evolution

        Args:
            rollout (torch.Tensor): Ground truth of the system evolution, concatenated along last axis
        """
        self.input = rollout

    def set_z(self, z_mean, z_std, z_sample):
        """Store latent variable conditions

        Args:
            z_mean (torch.Tensor): Mean of q_z
            z_std (torch.Tensor): Standard dev of q_z
            z_sample (torch.Tensor): Sample taken from q_z distribution
        """
        self.z_mean = z_mean
        self.z_std = z_std
        self.z_sample = z_sample

    def append_state(self, q, p):
        """Append the guessed position (q) and momentum (p) to guessed list 

        Args:
            q (torch.Tensor): Position encoded information
            p (torch.Tensor): Momentum encoded information
        """
        self.q_s.append(q)
        self.p_s.append(p)

    def append_reconstruction(self, reconstruction):
        """Append guessed reconstruction

        Args:
            reconstruction (torch.Tensor): Decoder of the HGN output
        """
        if self.reconstructed_rollout is None:
            self.reconstructed_rollout = reconstruction
        else:
            self.reconstructed_rollout = torch.cat(
                (self.reconstructed_rollout, reconstruction), dim=1)
