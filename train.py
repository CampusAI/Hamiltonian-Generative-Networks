"""Script to train the Hamiltonian Generative Network
"""
import os
import yaml

import numpy as np
import time
import torch
import tqdm

from utilities import debug_utils
from utilities.integrator import Integrator
from utilities.training_logger import TrainingLogger
from utilities.loader import load_hgn, get_online_dataloaders, get_offline_dataloaders


def train(params):
    """Instantiate and train the Hamiltonian Generative Network.

    Args:
        params (dict): Experiment parameters (see experiment_params folder).
    """
    # Set device and dtype TODO(oleguer): Make choosing cpu an option
    device = "cuda:" + str(
        params["gpu_id"]) if torch.cuda.is_available() else "cpu"
    dtype = torch.__getattribute__(params["networks"]["dtype"])

    # Load hgn from parameters to deice
    hgn = load_hgn(params=params, device=device, dtype=dtype)

    # Either generate data on-the-fly or load the data from disk
    if "environment" in params:
        print("Training with ONLINE data...")
        train_data_loader, test_data_loader = get_online_dataloaders(params)
    else:
        print("Training with OFFLINE data...")
        train_data_loader, test_data_loader = get_offline_dataloaders(params)

    # hgn.load(os.path.join(params["model_save_dir"], params["experiment_id"]))

    # Initialize training logger
    training_logger = TrainingLogger(hyper_params=params,
                                     loss_freq=100,
                                     rollout_freq=100,
                                     model_freq=10000)

    # Initialize tensorboard writer
    model_save_file = os.path.join(params["model_save_dir"],
                                   params["experiment_id"])

    # TRAIN
    for ep in range(params["optimization"]["epochs"]):
        print("Epoch %s / %s" %
              (str(ep + 1), str(params["optimization"]["epochs"])))
        pbar = tqdm.tqdm(train_data_loader)
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

    # TEST
    print("Testing...")
    test_error = 0
    pbar = tqdm.tqdm(test_data_loader)
    for _, rollout_batch in enumerate(pbar):
        rollout_batch = rollout_batch.to(device).type(dtype)
        prediction = hgn.forward(rollout_batch=rollout_batch,
                                 variational=params["networks"]["variational"])
        error = torch.nn.MSELoss()(
            input=prediction.input,
            target=prediction.reconstructed_rollout).detach().cpu().numpy()
        test_error += error / len(test_data_loader)
    training_logger.log_test_error(test_error)
    return hgn


if __name__ == "__main__":
    params_file = "experiment_params/default_online.yaml"

    # Read parameters
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Train HGN network
    hgn = train(params)