import torch

from networks.inference_net import EncoderNet, TransformerNet
from networks.hamiltonian_net import HamiltonianNet
from utilities.integrator import Integrator
from utilities.hgn_result import HgnResult


class HGN():
    """Hamiltonian Generative Network model.
    
    This class models the HGN and allows implements its training and evaluation.
    """
    def __init__(self,
                 encoder,
                 transformer,
                 hnn,
                 decoder,
                 integrator,
                 optimizer,
                 loss,
                 seq_len,
                 channels=3):
        """Instantiate a Hamiltonian Generative Network.

        Args:
            encoder (EncoderNet): Encoder neural network.
            transformer (TransformerNet): Transformer neural network.
            hnn (HamiltonianNet): Hamiltonian neural network.
            decoder (DecoderNet): Decoder neural network.
            integrator (Integrator): HGN integrator.
            optimizer (torch.optim): PyTorch Network optimizer.
            loss (torch.nn.modules.loss): PyTorch Loss.
            seq_len (int): Number of frames in each rollout.
            channels (int, optional): Number of channels of the images. Defaults to 3.
        """
        # Parameters
        self.seq_len = seq_len
        self.channels = channels

        # Modules
        self.encoder = encoder
        self.transformer = transformer
        self.hnn = hnn
        self.decoder = decoder
        self.integrator = integrator

        # Optimization
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, rollout, n_steps=None):
        """Get the prediction of the HGN for a given rollout of n_steps.

        Args:
            rollout (torch.Tensor(N, C, H, W)): Image sequence of the system evolution concatenated along the channels' axis.
            n_steps (integer, optional): Number of guessed steps, if None will match seq_len. Defaults to None.

        Returns:
            HgnResult: Bundle of the intermediate and final results of the HGN output.
        """
        assert (rollout.size()[1] == self.channels * self.seq_len)  # Wrong rollout channel dim
        n_steps = self.seq_len if n_steps is None else n_steps  # If n_steps not specified, match input sequence length

        prediction = HgnResult()
        prediction.set_input(rollout)

        # Latent distribution
        z, z_mean, z_std = self.encoder(rollout)
        prediction.set_z(z_mean=z_mean, z_std=z_std, z_sample=z)

        # Initial state
        q, p = self.transformer(z)
        prediction.append_state(q=q, p=p)

        # Initial state reconstruction
        x_reconstructed = self.decoder(q=q)
        prediction.append_reconstruction(x_reconstructed)

        # Estimate predictions
        for _ in range(n_steps - 1):
            # Compute next state
            q, p = self.integrator.step(q=q, p=p, hnn=self.hnn)
            prediction.append_state(q=q, p=p)

            # Compute state reconstruction
            x_reconstructed = self.decoder(q=q)
            prediction.append_reconstruction(x_reconstructed)
        return prediction

    def fit(self, rollouts):
        """Perform a training step with the given rollouts batch.

        Args:
            rollouts (torch.Tensor(N, C, H, W)): Image sequence of the system evolution concatenated along the channels' axis.

        Returns:
            float: Loss obtained forwarding the given rollouts batch.
        """
        self.optimizer.zero_grad()
        prediction = self.forward(rollout=rollouts)
        error = self.loss(input=prediction.input,
                          target=prediction.reconstructed_rollout)
        error.backward()
        # self.optimizer.step()
        return error

    def load(self, file_name):
        """Load networks' parameters

        Args:
            file_name (string): Path to the saved models
        """
        # TODO(oleguer): Load networks' parameters
        raise NotImplementedError

    def save(self, file_name):
        """Save networks' parameters

        Args:
            file_name (string): Path where to save the models
        """
        # TODO(oleguer): Save networks' parameters
        raise NotImplementedError
