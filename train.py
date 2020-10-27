"""Script to train the Hamiltonian Generative Network
"""
import ast
import argparse
import copy
import pprint
import os
import warnings
import yaml

import numpy as np
import torch
import tqdm

from utilities.integrator import Integrator
from utilities.training_logger import TrainingLogger
from utilities.loader import load_hgn, get_online_dataloaders, get_offline_dataloaders
from utilities.losses import reconstruction_loss, kld_loss, geco_constraint


def _avoid_overwriting(experiment_id):
    # This function throws an error if the given experiment data already exists in runs/
    logdir = os.path.join('runs', experiment_id)
    if os.path.exists(logdir):
        assert len(os.listdir(logdir)) == 0,\
            f'Experiment id {experiment_id} already exists in runs/. Remove it, change the name ' \
            f'in the yaml file.'


class HgnTrainer:
    def __init__(self, params, resume=False):
        """Instantiate and train the Hamiltonian Generative Network.

        Args:
            params (dict): Experiment parameters (see experiment_params folder).
        """

        self.params = params
        self.resume = resume

        if not resume:  # Fail if experiment_id already exist in runs/
            _avoid_overwriting(params["experiment_id"])

        # Set device
        self.device = params["device"]
        if "cuda" in self.device and not torch.cuda.is_available():
            warnings.warn(
                "Warning! Set to train in GPU but cuda is not available. Device is set to CPU.")
            self.device = "cpu"

        # Get dtype, will raise a 'module 'torch' has no attribute' if there is a typo
        self.dtype = torch.__getattribute__(params["networks"]["dtype"])

        # Load hgn from parameters to deice
        self.hgn = load_hgn(params=self.params,
                            device=self.device,
                            dtype=self.dtype)

        # Either generate data on-the-fly or load the data from disk
        if "train_data" in self.params["dataset"]:
            print("Training with OFFLINE data...")
            self.train_data_loader, self.test_data_loader = get_offline_dataloaders(self.params)
        else:
            print("Training with ONLINE data...")
            self.train_data_loader, self.test_data_loader = get_online_dataloaders(self.params)

        # Initialize training logger
        self.training_logger = TrainingLogger(
            hyper_params=self.params,
            loss_freq=100,
            rollout_freq=1000,
            model_freq=5000
        )

        # Initialize tensorboard writer
        self.model_save_file = os.path.join(
            self.params["model_save_dir"],
            self.params["experiment_id"]
        )

        # Define optimization modules
        optim_params = [
            {'params': self.hgn.encoder_q.parameters(),
                'lr': params["optimization"]["encoder_lr"]},
            {'params': self.hgn.transformer_q.parameters(),
                'lr': params["optimization"]["transformer_lr"]},
            {'params': self.hgn.hnn_q.parameters(),
                'lr': params["optimization"]["hnn_lr"]},
            {'params': self.hgn.encoder_p.parameters(),
                'lr': params["optimization"]["encoder_lr"]},
            {'params': self.hgn.transformer_p.parameters(),
                'lr': params["optimization"]["transformer_lr"]},
            {'params': self.hgn.hnn_p.parameters(),
                'lr': params["optimization"]["hnn_lr"]},
            {'params': self.hgn.decoder.parameters(),
                'lr': params["optimization"]["decoder_lr"]},
        ]
        self.optimizer = torch.optim.Adam(optim_params)

    def training_step(self, rollouts):
        """Perform a training step with the given rollouts batch.

        Args:
            rollouts (torch.Tensor): Tensor of shape (batch_size, seq_len, channels, height, width)
                corresponding to a batch of sampled rollouts.

        Returns:
            A dictionary of losses and the model's prediction of the rollout. The reconstruction loss and
            KL divergence are floats and prediction is the HGNResult object with data of the forward pass.
        """
        self.optimizer.zero_grad()

        if 'use_steps' in self.params['optimization']:
            nsteps = self.params['optimization']['use_steps']
            roll = rollouts[:, :nsteps]
        else:
            roll = rollouts
        hgn_output = self.hgn.forward(rollout_batch=roll)
        hgn_output.input = rollouts
        target = hgn_output.input
        prediction = hgn_output.reconstructed_rollout

        if self.params["networks"]["variational"]:
            tol = self.params["geco"]["tol"]
            alpha = self.params["geco"]["alpha"]
            lagrange_mult_param = self.params["geco"]["lagrange_multiplier_param"]

            C, rec_loss = geco_constraint(target, prediction, tol)  # C has gradient

            # Compute moving average of constraint C (without gradient)
            if self.C_ma is None:
                self.C_ma = C.detach()
            else:
                self.C_ma = alpha * self.C_ma + (1 - alpha) * C.detach()

            C = C + (self.C_ma - C.detach())  # Move C without affecting its gradient

            # Compute KL divergence
            mu_q = hgn_output.z_mean_q
            logvar_q = hgn_output.z_logvar_q
            kld_q = kld_loss(mu=mu_q, logvar=logvar_q)
            mu_p = hgn_output.z_mean_p
            logvar_p = hgn_output.z_logvar_p
            kld_p = kld_loss(mu=mu_p, logvar=logvar_p)

            kld = kld_q + kld_p

            # Compute losses
            train_loss = kld + self.langrange_multiplier * C

            # clamping the langrange multiplier to avoid inf values
            self.langrange_multiplier = self.langrange_multiplier * torch.exp(lagrange_mult_param * C.detach())
            self.langrange_multiplier = torch.clamp(self.langrange_multiplier,
                                                    1e-10, 1e10)

            losses = {
                'loss/train': train_loss.item(),
                'loss/kld': kld.item(),
                'loss/C': C.item(),
                'loss/C_ma': self.C_ma.item(),
                'loss/rec': rec_loss.item(),
                'other/langrange_mult': self.langrange_multiplier.item()
            }

        else:  # not variational
            # Compute frame reconstruction error
            train_loss = reconstruction_loss(
                target=prediction.input,
                prediction=prediction.reconstructed_rollout)
            losses = {'loss/train': train_loss.item()}

        train_loss.backward()
        self.optimizer.step()

        return losses, hgn_output

    def fit(self):
        """The trainer fits an HGN.
        """

        # Initial values for geco algorithm
        if self.params["networks"]["variational"]:
            self.langrange_multiplier = self.params["geco"]["initial_lagrange_multiplier"]
            self.C_ma = None

        # TRAIN
        for ep in range(self.params["optimization"]["epochs"]):
            print("Epoch %s / %s" % (str(ep + 1), str(self.params["optimization"]["epochs"])))
            pbar = tqdm.tqdm(self.train_data_loader)
            for batch_idx, rollout_batch in enumerate(pbar):
                # Move to device and change dtype
                rollout_batch = rollout_batch.to(self.device).type(self.dtype)

                # Do an optimization step
                losses, prediction = self.training_step(rollouts=rollout_batch)

                # Log progress
                self.training_logger.step(losses=losses,
                                          rollout_batch=rollout_batch,
                                          prediction=prediction,
                                          model=self.hgn)

                # Progress-bar msg
                msg = ", ".join([
                    f"{k}: {v:.2e}" for k, v in losses.items() if v is not None
                ])
                pbar.set_description(msg)
            # Save model
            self.hgn.save(self.model_save_file)

        self.test()
        return self.hgn

    def test(self):
        """Test after the training is finished.
        """
        print("Testing...")
        test_error = 0
        pbar = tqdm.tqdm(self.test_data_loader)
        for _, rollout_batch in enumerate(pbar):
            rollout_batch = rollout_batch.to(self.device).type(self.dtype)
            prediction = self.hgn.forward(rollout_batch=rollout_batch)
            error = reconstruction_loss(
                target=prediction.input,
                prediction=prediction.reconstructed_rollout).detach().cpu(
                ).numpy()
            test_error += error / len(self.test_data_loader)
        self.training_logger.log_test_error(test_error)

def _overwrite_config_with_cmd_arguments(config, args):
    if args.name is not None:
        config['experiment_id'] = args.name[0]
    if args.epochs is not None:
        config['optimization']['epochs'] = args.epochs[0]
    if args.dataset_path is not None:
        # Read the parameters.yaml file in the given dataset path
        dataset_config = _read_config(os.path.join(_args.dataset_path[0], 'parameters.yaml'))
        for key, value in dataset_config.items():
            config[key] = value
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


def _ask_confirmation(config):
    printer = pprint.PrettyPrinter(indent=4)
    print(f'The training will be run with the following configuration:')
    printed_config = copy.deepcopy(_config)
    printed_config.pop('networks')
    printer.pprint(printed_config)
    print('Proceed? (y/n):')
    if input() != 'y':
        print('Abort.')
        exit()


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
        '-y', '-y', action='store_true', default=False, required=False,
        help='Whether to skip asking for user confirmation before starting the training.'
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
        assert _args.dataset_config is None, 'Both --dataset-path and --dataset-config were given.'
        _config = _train_config

    # Overwrite configuration with command line arguments
    _overwrite_config_with_cmd_arguments(_config, _args)

    # Show configuration and ask user for confirmation
    if not _args.y:
        _ask_confirmation(_config)

    # Train HGN network
    trainer = HgnTrainer(_config)
    hgn = trainer.fit()
