import torch

from networks.inference_network import EncoderNet, TransformerNet
from networks.hamiltonian_net import HamiltonianNet
from utils.integrator import Integrator
from utils.hgn_result import HgnResult

class HGN():
    """Hamiltonian Generative Network model.
    
    This class models the HGN and allows implements its training and evaluation.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.encoder = EncoderNet(self.seq_len)
        self.transformer = TransformerNet()
        self.hnn = HamiltonianNet()
        self.decoder = None

        self.integrator = Integrator(delta_T=0.1, method="euler")
        

    def forward(self, rollout, steps=None):
        assert(rollout.size()[2] == 3*self.seq_len)  # Rollout channel dim needs to be 3*seq_len
        steps = self.seq_len if steps is None else steps  # If steps not specified, match input sequence length

        prediction = HgnResult()
        prediction.set_z(self.encoder(rollout))  # Latent distribution
        s_0 = self.transformer(prediction.z_sampled)  # Initial state
        prediction.set_initial_state(s_0)
        q, p = torch.split(s_0, split_size_or_sections=16, dim=2)  # Split into q, p
        prediction.add_step(q=q, p=p)
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
