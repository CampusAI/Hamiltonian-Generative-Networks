import torch

from networks.inference_net import EncoderNet, TransformerNet
from networks.hamiltonian_net import HamiltonianNet
from utils.integrator import Integrator
from utils.hgn_result import HgnResult


class HGN():
    """Hamiltonian Generative Network model.
    
    This class models the HGN and allows implements its training and evaluation.
    """
    def __init__(self, seq_len):
        self.seq_len = seq_len
        # TODO(oleguer): Pass network parameters
        self.encoder = EncoderNet(self.seq_len)
        self.transformer = TransformerNet()
        self.hnn = HamiltonianNet()
        self.decoder = None

        self.integrator = Integrator(delta_T=0.1, method="euler")

    def forward(self, rollout, steps=None):
        assert (rollout.size()[2] == 3 * self.seq_len)  # Rollout channel dim needs to be 3*seq_len
        steps = self.seq_len if steps is None else steps  # If steps not specified, match input sequence length

        prediction = HgnResult()

        # Latent distribution
        z, z_mean, z_std = self.encoder(rollout)
        prediction.set_z(z_mean=z_mean, z_std=z_std, z_sample=z)

        # Initial state
        q, p = self.transformer(prediction.z_sampled)
        prediction.add_step(q=q, p=p)

        # Estimate predictions
        for step in range(steps):
            q, p = Integrator.step(q=q, p=p, hnn=self.hnn)
            prediction.add_step(q=q, p=p)
        return prediction

    def train(self, rollouts):
        raise NotImplementedError

    def load(self, file_name):
        raise NotImplementedError

    def save(self, file_name):
        raise NotImplementedError
