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
    ENCODER_Q_FILENAME = "encoder_q.pt"
    TRANSFORMER_Q_FILENAME = "transformer_q.pt"
    HAMILTONIAN_Q_FILENAME = "hamiltonian_q.pt"
    DECODER_FILENAME = "decoder.pt"
    ENCODER_P_FILENAME = "encoder_p.pt"
    TRANSFORMER_P_FILENAME = "transformer_p.pt"
    HAMILTONIAN_P_FILENAME = "hamiltonian_p.pt"


    def __init__(self,
                 encoder_q,
                 transformer_q,
                 hnn_q,
                 encoder_p,
                 transformer_p,
                 hnn_p,
                 decoder,
                 integrator,
                 device,
                 dtype,
                 seq_len,
                 channels):
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
        self.encoder_q = encoder_q
        self.transformer_q = transformer_q
        self.hnn_q = hnn_q
        self.encoder_p = encoder_p
        self.transformer_p = transformer_p
        self.hnn_p = hnn_p
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
        prediction = HgnResult(batch_shape=torch.Size(prediction_shape),
                               device=self.device)
        prediction.set_input(rollout_batch)

        first_img = rollout_batch[:, 0]
        # Concat along channel dimension
        rollout_batch = conversions.concat_rgb(rollout_batch)

        # Latent distribution
        z_q, z_mean_q, z_logvar_q = self.encoder_q(first_img, sample=variational)
        z_p, z_mean_p, z_logvar_p = self.encoder_p(rollout_batch, sample=variational)
        prediction.set_z(z_sample_q=z_q, z_mean_q=z_mean_q, z_logvar_q=z_logvar_q,
                         z_sample_p=z_p, z_mean_p=z_mean_p, z_logvar_p=z_logvar_p)

        # Initial state
        q = self.transformer_q(z_q)
        p = self.transformer_p(z_p)
        prediction.append_state(q=q, p=p)

        # Initial state reconstruction
        x_reconstructed = self.decoder(q)
        prediction.append_reconstruction(x_reconstructed)

        # Estimate predictions
        for _ in range(n_steps - 1):
            # Compute next state
            q, p = self.integrator.step(q=q, p=p, hnn_q=self.hnn_q, hnn_p=self.hnn_p)
            prediction.append_state(q=q, p=p)
            prediction.append_energy(self.integrator.energy)  # This is the energy of previous timestep

            # Compute state reconstruction
            x_reconstructed = self.decoder(q)
            prediction.append_reconstruction(x_reconstructed)
        
        # We need to add the energy of the system at the last time-step
        with torch.no_grad():
            last_energy = self.hnn_q(x=q).detach().cpu().numpy() \
                          + self.hnn_p(x=p).detach().cpu().numpy()
        prediction.append_energy(last_energy)  # This is the energy of previous timestep
        return prediction

    def save(self, directory):
        """Save networks' parameters

        Args:
            directory (string): Path where to save the models, if does not exist it, is created
        """
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder_q, os.path.join(directory,
                                              self.ENCODER_Q_FILENAME))
        torch.save(self.transformer_q,
                   os.path.join(directory, self.TRANSFORMER_Q_FILENAME))
        torch.save(self.hnn_q, os.path.join(directory,
                                          self.HAMILTONIAN_Q_FILENAME))
        torch.save(self.decoder, os.path.join(directory,
                                              self.DECODER_FILENAME))
        torch.save(self.encoder_p, os.path.join(directory,
                                                self.ENCODER_P_FILENAME))
        torch.save(self.transformer_p,
                   os.path.join(directory, self.TRANSFORMER_P_FILENAME))
        torch.save(self.hnn_p, os.path.join(directory,
                                          self.HAMILTONIAN_P_FILENAME))


    def get_random_sample(self, n_steps, img_shape=(32, 32)):
        """Sample a rollout from the HGN

        Args:
            n_steps (int): Length of the sampled rollout
            img_shape (tuple(int, int), optional): Size of the images, should match the trained ones. Defaults to (32, 32).

        Returns:
            (utilities.HgnResult): An HgnResult object containing data of the forward pass over the
                given minibatch.
        """
        # Sample from a normal distribution the latent representation of the rollout
        latent_shape = (1, self.encoder.out_mean.out_channels, img_shape[0],
                        img_shape[1])
        latent_representation = torch.randn(latent_shape)

        # Instantiate prediction object
        prediction_shape = (1, n_steps, self.channels, img_shape[0],
                            img_shape[1])
        prediction = HgnResult(batch_shape=torch.Size(prediction_shape),
                               device=self.device)

        prediction.set_z(z_sample=latent_representation)

        # Initial state
        q, p = self.transformer(latent_representation)
        prediction.append_state(q=q, p=p)

        # Initial state reconstruction
        x_reconstructed = self.decoder(q)
        print(x_reconstructed.shape)
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
