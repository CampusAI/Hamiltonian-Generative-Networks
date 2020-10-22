"""Script to train the Hamiltonian Generative Network
"""
import argparse
import os
import yaml

import numpy as np
import time
import torch
import tqdm

from utilities.integrator import Integrator
from utilities.training_logger import TrainingLogger
from utilities.loader import load_hgn, get_online_dataloaders, get_offline_dataloaders


def _avoid_overwriting(experiment_id):
    # This function throws an error if the given experiment data already exists in runs/
    logdir = os.path.join('runs', experiment_id)
    if os.path.exists(logdir):
        assert len(os.listdir(logdir)) == 0,\
            f'Experiment id {experiment_id} already exists in runs/. Remove it, change the name ' \
            f'in the yaml file.'

def train(params, cpu=False, resume=False):
    """Instantiate and train the Hamiltonian Generative Network.

    Args:
        params (dict): Experiment parameters (see experiment_params folder).
    """

    # Set device
    device = 'cpu'
    # "cuda:" + str(
        # params["gpu_id"]) if torch.cuda.is_available() else "cpu"

    # Get dtype, will raise a 'module 'torch' has no attribute' if there is a typo
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


    # Initialize training logger
    training_logger = TrainingLogger(hyper_params=params,
                                     loss_freq=1,
                                     rollout_freq=10,
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

    DEFAULT_PARAM_FILE = "experiment_params/default_online.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', action='store', nargs=1, required=False,
        help='Path to the yaml file with the training configuration. If not specified,'
             'experiment_params/default_online.yaml will be used'
    )
    parser.add_argument(
        '--name', action='store', nargs=1, required=False,
        help='If specified, this name will be used instead of experiment_name of the yaml file.'
    )
    parser.add_argument(
        '--cpu', action='store_true', required=False, default=False,
        help='If specified, the training will be run on cpu. Otherwise, it will be run on GPU, '
             'unless GPU is not available.'
    )
    parser.add_argument(
        '--resume', action='store', required=False, nargs='?', default=None,
        help='Resume the training from a saved model. If a path is provided, the training will '
             'be resumed from the given checkpoint. Otherwise, the last checkpoint will be taken '
             'from saved_models/<experiment_id>'
    )
    args = parser.parse_args()

    if args.resume is not None:
        raise NotImplementedError('Resume training from command line is not implemented yet')

    params_file = args.params[0] if args.params is not None else DEFAULT_PARAM_FILE
    # Read parameters
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if args.name is not None:
        params['experiment_id'] = args.name[0]
    # Train HGN network
    hgn = train(params, cpu=args.cpu)
