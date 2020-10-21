import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.datasets import EnvironmentSampler, EnvironmentLoader
from environments.environment_factory import EnvFactory
from hamiltonian_generative_network import HGN
from networks.decoder_net import DecoderNet
from networks.encoder_net import EncoderNet
from networks.hamiltonian_net import HamiltonianNet
from networks.transformer_net import TransformerNet
from utilities.integrator import Integrator


def load_hgn(params, device, dtype):
    """Return the Hamiltonian Generative Network created from the given parameters.

    Args:
        params (dict): Experiment parameters (see experiment_params folder).
        device (str): String with the device to use. E.g. 'cuda:0', 'cpu'.
        dtype (torch.dtype): Data type to be used by the networks.
    """
    # Define networks
    encoder = EncoderNet(seq_len=params["rollout"]["seq_length"],
                         in_channels=params["rollout"]["n_channels"],
                         **params["networks"]["encoder"],
                         dtype=dtype).to(device)
    transformer = TransformerNet(
        in_channels=params["networks"]["encoder"]["out_channels"],
        **params["networks"]["transformer"],
        dtype=dtype).to(device)
    hnn = HamiltonianNet(**params["networks"]["hamiltonian"],
                         dtype=dtype).to(device)
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"],
        out_channels=params["rollout"]["n_channels"],
        **params["networks"]["decoder"],
        dtype=dtype).to(device)

    # Define HGN integrator
    integrator = Integrator(delta_t=params["rollout"]["delta_time"],
                            method=params["integrator"]["method"])
    # Define optimization modules
    optim_params = [
        {
            'params': encoder.parameters(),
            'lr': params["optimization"]["encoder_lr"]
        },
        {
            'params': transformer.parameters(),
            'lr': params["optimization"]["transformer_lr"]
        },
        {
            'params': hnn.parameters(),
            'lr': params["optimization"]["hnn_lr"]
        },
        {
            'params': decoder.parameters(),
            'lr': params["optimization"]["decoder_lr"]
        },
    ]
    optimizer = torch.optim.Adam(optim_params)
    loss = torch.nn.MSELoss()
    # Instantiate Hamiltonian Generative Network
    hgn = HGN(encoder=encoder,
              transformer=transformer,
              hnn=hnn,
              decoder=decoder,
              integrator=integrator,
              loss=loss,
              optimizer=optimizer,
              device=device,
              dtype=dtype,
              seq_len=params["rollout"]["seq_length"],
              channels=params["rollout"]["n_channels"])
    return hgn


def get_online_dataloaders(params):
    """Get train and test online environment dataloaders for the given params

    Args:
        params (dict): Experiment parameters (see experiment_params folder).

    Returns:
        tuple(torch.utils.data.DataLoader, torch.utils.data.DataLoader): Train and test dataloader
    """
    # Pick environment
    env = EnvFactory.get_environment(**params["environment"])

    # Train
    trainDS = EnvironmentSampler(
        environment=env,
        dataset_len=params["dataset"]["num_train_samples"],
        number_of_frames=params["rollout"]["seq_length"],
        delta_time=params["rollout"]["delta_time"],
        number_of_rollouts=params["optimization"]["batch_size"],
        img_size=params["dataset"]["img_size"],
        color=params["rollout"]["n_channels"] == 3,
        noise_std=params["dataset"]["noise_std"],
        radius_bound=params["dataset"]["radius_bound"],
        world_size=params["dataset"]["world_size"],
        seed=None)
    train_data_loader = torch.utils.data.DataLoader(trainDS,
                                                    shuffle=False,
                                                    batch_size=None)
    # Test
    testDS = EnvironmentSampler(
        environment=env,
        dataset_len=params["dataset"]["num_test_samples"],
        number_of_frames=params["rollout"]["seq_length"],
        delta_time=params["rollout"]["delta_time"],
        number_of_rollouts=params["optimization"]["batch_size"],
        img_size=params["dataset"]["img_size"],
        color=params["rollout"]["n_channels"] == 3,
        noise_std=params["dataset"]["noise_std"],
        radius_bound=params["dataset"]["radius_bound"],
        world_size=params["dataset"]["world_size"],
        seed=None)
    test_data_loader = torch.utils.data.DataLoader(testDS,
                                                   shuffle=False,
                                                   batch_size=None)
    return train_data_loader, test_data_loader


def get_offline_dataloaders(params):
    """Get train and test online environment dataloaders for the given params

    Args:
        params (dict): Experiment parameters (see experiment_params folder).

    Returns:
        tuple(torch.utils.data.DataLoader, torch.utils.data.DataLoader): Train and test dataloader
    """
    # Train
    trainDS = EnvironmentLoader(params["dataset"]["train_data"])
    train_data_loader = torch.utils.data.DataLoader(
        trainDS, shuffle=True, batch_size=params["optimization"]["batch_size"])

    # Test
    test_DS = EnvironmentLoader(params["dataset"]["test_data"])
    test_data_loader = torch.utils.data.DataLoader(
        test_DS, shuffle=True, batch_size=params["optimization"]["batch_size"])

    return train_data_loader, test_data_loader