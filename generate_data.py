import argparse
import ast
import copy
import shutil
import sys
import os
import yaml

import numpy as np
from tqdm import tqdm

from environments import environment_factory


def generate_and_save(root_path, environment, n_samples, n_frames, delta_time, img_size,
                      radius_bound, noise_level, color, start_seed, train=True):
    path = os.path.join(root_path, 'train' if train else 'test')
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(range(n_samples)):
        rolls = environment.sample_random_rollouts(
            number_of_frames=n_frames,
            delta_time=delta_time,
            number_of_rollouts=1,
            img_size=img_size,
            noise_level=noise_level,
            radius_bound=radius_bound,
            color=color,
            seed=i + start_seed
        )[0]
        filename = "{0:05d}".format(i)
        np.savez(os.path.join(path, filename), rolls)
    return path


def _read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def _prepare_out_config(config, train_path, test_path):
    out_config = copy.deepcopy(config)
    out_config['dataset']['train_data'] = train_path
    out_config['dataset']['test_data'] = test_path
    return out_config


def _overwrite_config_with_cmd_arguments(config, args):
    # This function overwrites parameters in the given dictionary
    # with the correspondent command line arguments.
    if args.ntrain is not None:
        config['dataset']['num_train_samples'] = args.ntrain[0]
    if args.ntest is not None:
        config['dataset']['num_test_samples'] = args.ntest[0]
    if args.env is not None:
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


if __name__ == '__main__':
    DEFAULT_DATASETS_ROOT = 'datasets/'
    DEFAULT_DATASET_CONFIG_FILE = 'experiment_params/dataset_online_default.yaml'
    DEFAULT_TRAIN_CONFIG_FILE = 'experiment_params/train_config_default.yaml'
    DEFAULT_ENVIRONMENTS_PATH = 'experiment_params/default_environments/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', action='store', nargs=1, required=True, help='The dataset name.'
    )
    parser.add_argument(
        '--dataset-config', action='store', nargs=1, type=str, required=False,
        help=f'YAML file from which to read the dataset parameters. If not specified, '
             f'{DEFAULT_DATASET_CONFIG_FILE} will be used.'
    )
    parser.add_argument(
        '--ntrain', action='store', nargs=1, required=False, type=int,
        help='Number of training sample to generate.'
    )
    parser.add_argument(
        '--ntest', action='store', nargs=1, required=False, type=int,
        help='Number of test samples to generate.'
    )
    parser.add_argument(
        '--env', action='store', nargs=1, required=False, type=str,
        help=f'The default environment specifications to use. Can be \'pendulum\', \'spring\', '
             f'\'two_bodies\', \'three_bodies\', \'chaotic_pendulum\'. If this argument is '
             f'specified, a default environment section will be loaded from the correspondent yaml '
             f'file in {DEFAULT_ENVIRONMENTS_PATH}'
    )
    parser.add_argument(
        '--datasets-root', action='store', nargs=1, required=False, type=str,
        help=f'Root of the datasets folder in which the dataset will be stored. If not specified, '
             f'{DEFAULT_DATASETS_ROOT} will be used as default.'
    )
    parser.add_argument(
        '--params', action='store', nargs='+', required=False,
        help='Override one or more parameters in the config. The format of an argument is '
             'param_name=param_value. Nested parameters are accessible by using a dot, '
             'i.e. --param dataset.img_size=32. IMPORTANT: lists must be enclosed in double '
             'quotes, i.e. --param environment.mass:"[0.5, 0.5]".'
    )
    _args = parser.parse_args()

    # Read yaml file with parameters definition
    _dataset_config_file = _args.dataset_config[0] if _args.dataset_config is not None else \
        DEFAULT_DATASET_CONFIG_FILE
    _dataset_config = _read_config(_dataset_config_file)

    # Overwrite dictionary from command line args to ensure they will be used
    _overwrite_config_with_cmd_arguments(_dataset_config, _args)

    # Extract environment parameters
    EXP_NAME = _args.name[0]
    N_TRAIN_SAMPLES = _dataset_config['dataset']['num_train_samples']
    N_TEST_SAMPLES = _dataset_config['dataset']['num_test_samples']
    IMG_SIZE = _dataset_config['dataset']['img_size']
    RADIUS_BOUND = _dataset_config['dataset']['radius_bound']
    NOISE_LEVEL = _dataset_config['dataset']['rollout']['noise_level']
    N_FRAMES = _dataset_config['dataset']['rollout']['seq_length']
    DELTA_TIME = _dataset_config['dataset']['rollout']['delta_time']
    N_CHANNELS = _dataset_config['dataset']['rollout']['n_channels']

    # Get dataset output path
    dataset_root = DEFAULT_DATASETS_ROOT if _args.datasets_root is None else _args.datasets_root[0]
    dataset_root = os.path.join(dataset_root, EXP_NAME)

    # Get the environment object from dictionary parameters
    environment = environment_factory.EnvFactory.get_environment(**_dataset_config['environment'])

    # Ask user confirmation
    print(f'The dataset will be generated with the following configuration:')
    print(f'PATH: {dataset_root}')
    print(f'dataset: {_dataset_config["dataset"]}')
    print(f'environment: {_dataset_config["environment"]}')
    print('\nProceed? (y/n):')
    if input() != 'y':
        print('Aborting')
        exit()

    # Generate train samples
    _train_path = generate_and_save(
        root_path=dataset_root, environment=environment,
        n_samples=N_TRAIN_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME, img_size=IMG_SIZE,
        radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL, color=N_CHANNELS == 3,
        start_seed=0, train=True
    )

    # Generate test samples
    _test_path = None
    if N_TEST_SAMPLES > 0:
        _test_path = generate_and_save(
            root_path=dataset_root, environment=environment,
            n_samples=N_TEST_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME,
            img_size=IMG_SIZE, radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL,
            color=N_CHANNELS == 3, start_seed=N_TRAIN_SAMPLES, train=False
        )

    # Convert parameters to offline train parameters and write them in the dataset
    _out_config = _prepare_out_config(_dataset_config, _train_path, _test_path)
    yaml_content = yaml.dump(_out_config, default_flow_style=True)
    config_out_path = os.path.join(dataset_root, 'parameters.yaml')
    with open(config_out_path, 'x') as f:
        f.write(yaml_content)
        f.close()

    print(f'A parameter file ready to be trained on was generated at {config_out_path}')
