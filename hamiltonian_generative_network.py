import os
import pathlib

import torch

from utilities import conversions
from utilities.integrator import Integrator
from utilities.hgn_result import HgnResult


class HGN:
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
                 device,
                 dtype,
                 seq_len,
                 channels=3):
        """Instantiate a Hamiltonian Generative Network.

        Args:
            encoder (networks.encoder_net.EncoderNet): Encoder neural network.
            transformer (networks.transformer_net.TransformerNet): Transformer neural network.
            hnn (networks.hamiltonian_net.HamiltonianNet): Hamiltonian neural network.
            decoder (networks.decoder_net.DecoderNet): Decoder neural network.
            integrator (Integrator): HGN integrator.
            device (str): String with the device to use. E.g. 'cuda:0', 'cpu'.
            dtype (torch.dtype): Data type used for the networks.
            seq_len (int): Number of frames in each rollout.
            channels (int, optional): Number of channels of the images. Defaults to 3.
        """
        # Parameters
        self.seq_len = seq_len
        self.channels = channels
        self.device = device
        self.dtype = dtype
        
        # Modules
        self.encoder = encoder
        self.transformer = transformer
        self.hnn = hnn
        self.decoder = decoder
        self.integrator = integrator
    
    def forward(self, rollout_batch, n_steps=None, variational=True):
        """Get the prediction of the HGN for a given rollout_batch of n_steps.

        Args:
            rollout_batch (torch.Tensor): Minibatch of rollouts as a Tensor of shape
                (batch_size, seq_len, channels, height, width).
            n_steps (integer, optional): Number of guessed steps, if None will match seq_len.
                Defaults to None.
            variational (bool): Whether to sample from the encoder distribution or take the mean.

        Returns:
            (utilities.HgnResult): An HgnResult object containing data of the forward pass over the
                given minibatch.
        """
        n_steps = self.seq_len if n_steps is None else n_steps

        # Instantiate prediction object
        prediction_shape = list(rollout_batch.shape)
        prediction_shape[1] = n_steps
        prediction = HgnResult(batch_shape=torch.Size(prediction_shape), device=self.device)
        prediction.set_input(rollout_batch)

        # Concat along channel dimension
        rollout_batch = conversions.concat_rgb(rollout_batch)

        # Latent distribution
        z, z_mean, z_logvar = self.encoder(rollout_batch, sample=variational)
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

    def debug_mode(self):
        """Set the network to debug mode, i.e. allow intermediate gradients to be retrieved.
        """
        for module in [self.encoder, self.transformer, self.decoder, self.hnn]:
            for name, layer in module.named_parameters():
                layer.retain_grad()
