import os
import pathlib

import torch

from networks.inference_net import concat_rgb
from utilities.integrator import Integrator
from utilities.hgn_result import HgnResult

class HGN():
    """Hamiltonian Generative Network model.
    
    This class models the HGN and implements its training and evaluation.
    """
    ENCODER_FILENAME = "encoder.pt"
    TRANSFORMER_FILENAME = "transformer.pt"
    HAMILTONIAN_FILENAME = "hamiltonian.pt"
    DECODER_FILENAME = "decoder.pt"

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

    def forward(self, rollout_batch, n_steps=None):
        """Get the prediction of the HGN for a given rollout_batch of n_steps.

        Args:
            rollout_batch (torch.Tensor(N, C, H, W)): Image sequence of the system evolution concatenated along the channels' axis.
            n_steps (integer, optional): Number of guessed steps, if None will match seq_len. Defaults to None.

        Returns:
            HgnResult: Bundle of the intermediate and final results of the HGN output.
        """
        rollout_batch = concat_rgb(rollout_batch)
        assert (rollout_batch.size()[1] == self.channels * self.seq_len)  # Wrong rollout_batch channel dim
        n_steps = self.seq_len if n_steps is None else n_steps  # If n_steps not specified, match input sequence length

        prediction = HgnResult()
        prediction.set_input(rollout_batch)

        # Latent distribution
        z, z_mean, z_logvar = self.encoder(rollout_batch)
        prediction.set_z(z_mean=z_mean, z_logvar=z_logvar, z_sample=z)

        # Initial state
        q, p = self.transformer(z)
        prediction.append_state(q=q, p=p)

        # Initial state reconstruction
        x_reconstructed = self.decoder(q)
        prediction.append_reconstruction(x_reconstructed)

        # Estimate predictions
        for _ in range(n_steps - 1):
            # Compute next state
            q, p = self.integrator.step(q=q, p=p, hnn=self.hnn)
            prediction.append_state(q=q, p=p)

            # Compute state reconstruction
            x_reconstructed = self.decoder(q)
            prediction.append_reconstruction(x_reconstructed)
        return prediction

    def fit(self, rollouts):
        """Perform a training step with the given rollouts batch.

        Args:
            rollouts (torch.Tensor(N, C, H, W)): Image sequence of the system evolution concatenated along the channels' axis.

        Returns:
            float: Loss obtained forwarding the given rollouts batch.
        """
        # Re-set gradients and forward new batch
        self.optimizer.zero_grad()
        prediction = self.forward(rollout_batch=rollouts)
        
        # Compute frame reconstruction error
        reconstruction_error = self.loss(input=prediction.input,
                                target=prediction.reconstructed_rollout)
    
        # Compute KL divergence
        mu = prediction.z_mean
        logvar = prediction.z_logvar
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # NOTE(oleguer): Sum or mean?

        # Compute loss
        beta = 0.  #TODO(Stathi) Compute beta value
        error = reconstruction_error + beta*KLD
        
        # Optimization step
        error.backward()
        self.optimizer.step()
        return float(reconstruction_error.detach().cpu().numpy()), float(KLD.detach().cpu().numpy()), prediction

    def load(self, directory):
        """Load networks' parameters

        Args:
            directory (string): Path to the saved models
        """
        self.encoder = torch.load(os.path.join(directory, self.ENCODER_FILENAME))
        self.transformer = torch.load(os.path.join(directory, self.TRANSFORMER_FILENAME))
        self.hnn = torch.load(os.path.join(directory, self.HAMILTONIAN_FILENAME))
        self.decoder = torch.load(os.path.join(directory, self.DECODER_FILENAME))

    def save(self, directory):
        """Save networks' parameters

        Args:
            directory (string): Path where to save the models, if does not exist it, is created
        """
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder, os.path.join(directory, self.ENCODER_FILENAME))
        torch.save(self.transformer, os.path.join(directory, self.TRANSFORMER_FILENAME))
        torch.save(self.hnn, os.path.join(directory, self.HAMILTONIAN_FILENAME))
        torch.save(self.decoder, os.path.join(directory, self.DECODER_FILENAME))
