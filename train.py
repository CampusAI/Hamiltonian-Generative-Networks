"""Script to train the Hamiltonian Generative Network
"""
import ast
import argparse
import copy
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
    if not resume:
        _avoid_overwriting(params['experiment_id'])  # Avoid overwriting tensorboard data
    # Set device and dtype
    if cpu:
        device = 'cpu'
    else:
        device = "cuda:" + str(
            params["gpu_id"]) if torch.cuda.is_available() else "cpu"
    dtype = torch.__getattribute__(params["networks"]["dtype"])

    # Load hgn from parameters to deice
    hgn = load_hgn(params=params, device=device, dtype=dtype)

    # Either generate data on-the-fly or load the data from disk
    if "train_data" in params["dataset"]:
        print(f"Training with OFFLINE data from dataset {params['dataset']['train_data']}")
        train_data_loader, test_data_loader = get_offline_dataloaders(params)
    else:
        assert "environment" in params, "Nor environment nor train_data are specified in the " \
                                        "given configuration."
        print("Training with ONLINE data...")
        train_data_loader, test_data_loader = get_online_dataloaders(params)

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


def _overwrite_config_with_cmd_arguments(config, args):
    if args.name is not None:
        config['experiment_id'] = args.name[0]
    if args.epochs is not None:
        config['optimization']['epochs'] = args.epochs[0]
    if args.dataset_path is not None:
        # Read the parameters.yaml file in the given dataset path
        dataset_config = _read_config(os.path.join(_args.dataset_path[0], 'parameters.yaml'))
        config['dataset'] = {
            'train_data': dataset_config['dataset']['train_data'],
            'test_data': dataset_config['dataset']['test_data']
        }
        config['dataset']['rollout'] = dataset_config['dataset']['rollout']
    if args.env is not None:
        if 'train_data' in config['dataset']:
            raise ValueError(
                f'--env was given but configuration is set for offline training: '
                f'train_data={config["dataset"]["train_data"]}'
            )
        env_params = _read_config(DEFAULT_ENVIRONMENTS_PATH + args.env[0] + '.yaml')
        config['environment'] = env_params['environment']
    if args.params is not None:
        for p in args.params:
            key, value = p.split('=')
            ptr = config
            keys = key.split('.')
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    ptr[k] = ast.literal_eval(value)
                else:
                    ptr = ptr[k]


def _read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _merge_configs(train_config, dataset_config):
    config = copy.deepcopy(train_config)
    for key, value in dataset_config.items():
        config[key] = value
    # If the config specifies a dataset path, we take the rollout from the configuration file
    # in the given dataset
    if 'dataset' in config and 'train_data' in config['dataset']:
        dataset_config = _read_config(  # Read parameters.yaml in root of given dataset
            os.path.join(os.path.dirname(config['dataset']['train_data']), 'parameters.yaml'))
        config['dataset']['rollout'] = dataset_config['dataset']['rollout']
    return config


if __name__ == "__main__":

    DEFAULT_TRAIN_CONFIG_FILE = "experiment_params/train_config_default.yaml"
    DEFAULT_DATASET_CONFIG_FILE = "experiment_params/dataset_online_default.yaml"
    DEFAULT_ENVIRONMENTS_PATH = "experiment_params/default_environments/"
    DEFAULT_SAVE_MODELS_DIR = "saved_models/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-config', action='store', nargs=1, type=str, required=True,
        help=f'Path to the training configuration yaml file.'
    )
    parser.add_argument(
        '--dataset-config', action='store', nargs=1, type=str, required=False,
        help=f'Path to the dataset configuration yaml file.'
    )
    parser.add_argument(
        '--name', action='store', nargs=1, required=False,
        help='If specified, this name will be used instead of experiment_id of the yaml file.'
    )
    parser.add_argument(
        '--epochs', action='store', nargs=1, type=int, required=False,
        help='The number of training epochs. If not specified, optimization.epochs of the '
             'training configuration will be used.'
    )
    parser.add_argument(
        '--env', action='store', nargs=1, type=str, required=False,
        help='The environment to use (for online training only). Possible values are '
             '\'pendulum\', \'spring\', \'two_bodies\', \'three_bodies\', corresponding to '
             'environment configurations in experiment_params/default_environments/. If not '
             'specified, the environment specified in the given --dataset-config will be used.'
    )
    parser.add_argument(
        '--dataset-path', action='store', nargs=1, type=str, required=False,
        help='Path to a stored dataset to use for training. For offline training only. In this '
             'case no dataset configuration file will be loaded.'
    )
    parser.add_argument(
        '--params', action='store', nargs='+', required=False,
        help='Override one or more parameters in the config. The format of an argument is '
             'param_name=param_value. Nested parameters are accessible by using a dot, '
             'i.e. --param dataset.img_size=32. IMPORTANT: lists must be enclosed in double '
             'quotes, i.e. --param environment.mass:"[0.5, 0.5]".'
    )
    parser.add_argument(
        '--cpu', action='store_true', required=False, default=False,
        help='If specified, the training will be run on cpu. Otherwise, it will be run on GPU, '
             'unless GPU is not available.'
    )
    parser.add_argument(
        '--resume', action='store', required=False, nargs='?', default=None,
        help='NOT IMPLEMENTED YET. Resume the training from a saved model. If a path is provided, '
             'the training will be resumed from the given checkpoint. Otherwise, the last '
             'checkpoint will be taken from saved_models/<experiment_id>.'
    )
    _args = parser.parse_args()

    # Read configurations
    _train_config = _read_config(_args.train_config[0])
    if _args.dataset_path is None:  # Will use the dataset config file (or default if not given)
        _dataset_config_file = DEFAULT_DATASET_CONFIG_FILE if _args.dataset_config is None else \
            _args.dataset_config[0]
        _dataset_config = _read_config(_dataset_config_file)
        _config = _merge_configs(_train_config, _dataset_config)
    else:  # Will use the dataset given in the command line arguments
        _config = _train_config

    # Overwrite configuration with command line arguments
    _overwrite_config_with_cmd_arguments(_config, _args)

    # Train HGN network
    hgn = train(_config, cpu=_args.cpu)
