"""Script to train the Hamiltonian Generative Network
"""
import os
import yaml

import numpy as np
import time
import torch
import tqdm

from environments.datasets import EnvironmentSampler, EnvironmentLoader
from environments.environment import visualize_rollout
from environments.environment_factory import EnvFactory
from hamiltonian_generative_network import HGN
from networks.decoder_net import DecoderNet
from networks.encoder_net import EncoderNet
from networks.hamiltonian_net import HamiltonianNet
from networks.transformer_net import TransformerNet
from utilities.integrator import Integrator
from utilities.training_logger import TrainingLogger
from utilities import debug_utils


def load_hgn(params, device, dtype):
    """Return the Hamiltonian Generative Network created from the given parameters.

    Args:
        params (dict): Experiment parameters (see experiment_params folder).
        device (str): String with the device to use. E.g. 'cuda:0', 'cpu'.
    """
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
              seq_len=params["rollout"]["seq_length"],
              channels=params["rollout"]["n_channels"])
    return hgn


def train(params):
    """Instantiate and train the Hamiltonian Generative Network.

    Args:
        params (dict): Experiment parameters (see experiment_params folder).
    """
    # Set device
    device = "cuda:" + str(
        params["gpu_id"]) if torch.cuda.is_available() else "cpu"

    # Get dtype, will raise a 'module 'torch' has no attribute' if there is a typo
    dtype = torch.__getattribute__(params["networks"]["dtype"])

    # Pick environment
    env = EnvFactory.get_environment(**params["environment"])

    # Load hgn from parameters to deice
    hgn = load_hgn(params=params, device=device, dtype=dtype)

    # Either generate data on-the-fly or load the data from disk
    data_loader = None
    if params["dataset"]["on_the_fly_data"]:
        dataset_len = params["optimization"]["epochs"] * params[
            "optimization"]["batch_size"]
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
            seed=None)

        data_loader = torch.utils.data.DataLoader(
            trainDS,
            shuffle=False,
            batch_size=None)
    else:
        trainDS = EnvironmentLoader(params["dataset"]["train_data"])

        # Dataloader instance test, batch_mode enabled
        data_loader = torch.utils.data.DataLoader(
            trainDS,
            shuffle=True,
            batch_size=params["optimization"]["batch_size"])

    # hgn.load(os.path.join(params["model_save_dir"], params["experiment_id"]))
    training_logger = TrainingLogger(hyper_params=params,
                                     loss_freq=100,
                                     rollout_freq=1000,
                                     model_freq=1000)

    # Initialize tensorboard writer
    model_save_file = os.path.join(params["model_save_dir"],
                                   params["experiment_id"])
    for ep in range(params["optimization"]["epochs"]):
        print("Epoch %s / %s" % (str(ep), str(params["optimization"]["epochs"])))
        pbar = tqdm.tqdm(data_loader)
        for _, rollout_batch in enumerate(pbar):
            # Move to device and change dtype
            rollout_batch = rollout_batch.to(device).type(dtype)

            # Do an optimization step
            error, kld, prediction = hgn.fit(
                rollouts=rollout_batch,
                variational=params["networks"]["variational"])

            # Log progress
            training_logger.step(losses=(error, kld),
                                 rollout_batch=rollout_batch,
                                 prediction=prediction,
                                 model=hgn)

            # Progress-bar msg
            msg = "Loss: %s" % np.round(error, 5)
            if kld is not None:
                msg += ", KL: %s" % np.round(kld, 5)
            pbar.set_description(msg)
        # Save model
        hgn.save(model_save_file)
    return hgn

# def test(hgn, params):
#     test_DS = EnvironmentLoader(params["dataset"]["test_data"])
#     data_loader = torch.utils.data.DataLoader(
#         test_DS,
#         shuffle=False,
#         batch_size=1)

#     pbar = tqdm.tqdm(data_loader)
#     for _, rollout_batch in enumerate(pbar):
#         rollout_batch = rollout_batch.to(device).type(dtype)
#         result = hgn.forward()



if __name__ == "__main__":
    params_file = "experiment_params/default.yaml"

    # Read parameters
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Train HGN network
    hgn = train(params)
    # test(hgn, params)