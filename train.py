"""Script to train the Hamiltonian Generative Network
"""
import os

import time
import torch
import tqdm
import yaml
import numpy as np

from environments.datasets import EnvironmentSampler
from environments.environment import visualize_rollout
from environments.environment_factory import EnvFactory
from hamiltonian_generative_network import HGN
from networks.inference_net import EncoderNet, TransformerNet
from networks.hamiltonian_net import HamiltonianNet
from networks.decoder_net import DecoderNet
from utilities.integrator import Integrator
from utilities.training_logger import TrainingLogger


def train(params, dtype=torch.float):
    """Instantiate and train the Hamiltonian Generative Network.

    Args:
        params (dict): Experiment parameters (see experiment_params folder)
    """
    # Set device
    device = "cuda:" + str(
        params["gpu_id"]) if torch.cuda.is_available() else "cpu"

    # Pick environment
    env = EnvFactory.get_environment(**params["environment"])

    # Instantiate networks
    encoder = EncoderNet(
        seq_len=params["rollout"]["seq_length"], in_channels=params["rollout"]["n_channels"],
        **params["networks"]["encoder"], dtype=dtype).to(device)
    transformer = TransformerNet(
        in_channels=params["networks"]["encoder"]["out_channels"],
        **params["networks"]["transformer"], dtype=dtype).to(device)
    hnn = HamiltonianNet(**params["networks"]["hamiltonian"], dtype=dtype).to(device)
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"],
        out_channels=params["rollout"]["n_channels"], **params["networks"]["decoder"],
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
              seq_len=params["rollout"]["seq_length"],
              channels=params["rollout"]["n_channels"])

    # Dataloader
    dataset_len = params["optimization"]["epochs"] * params["optimization"]["batch_size"]
    seed = None if params["dataset"]["random"] else 0
    trainDS = EnvironmentSampler(
        environment=env,
        dataset_len=dataset_len,
        number_of_frames=params["rollout"]["seq_length"],
        delta_time=params["rollout"]["delta_time"],
        number_of_rollouts=params["optimization"]["batch_size"],
        img_size=params["dataset"]["img_size"],
        color=params["rollout"]["n_channels"] == 3,
        noise_std=params["dataset"]["noise_std"],
        radius_bound=params["dataset"]["radius_bound"],
        world_size=params["dataset"]["world_size"],
        seed=seed,
        dtype=dtype)
    # Dataloader instance test, batch_mode disabled
    data_loader = torch.utils.data.DataLoader(trainDS,
                                              shuffle=False,
                                              batch_size=None)

    # hgn.load(os.path.join(params["model_save_dir"], params["experiment_id"]))
    training_logger = TrainingLogger(hyper_params=params,
                                     loss_freq=100,
                                     rollout_freq=100)

    # Initialize tensorboard writer
    pbar = tqdm.tqdm(data_loader)
    for i, rollout_batch in enumerate(pbar):
        # rollout_batch has shape (batch_len, seq_len, channels, height, width)
        rollout_batch = rollout_batch.to(device)
        error, kld, prediction = hgn.fit(rollout_batch)
        training_logger.step(losses=(error, kld),
                             rollout_batch=rollout_batch,
                             prediction=prediction)
        msg = "Loss: %s, KL: %s" % (np.round(error, 4), np.round(kld, 4))
        pbar.set_description(msg)
    hgn.save(os.path.join(params["model_save_dir"], params["experiment_id"]))


if __name__ == "__main__":
    params_file = "experiment_params/default.yaml"
    
    # Read parameters
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Train HGN network
    train(params, dtype=torch.double)
