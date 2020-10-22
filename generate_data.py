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


def _read_params(params_file):
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def _prepare_out_params(params, train_path, test_path):
    offline_params = copy.deepcopy(params)
    offline_params.pop('img_size', None)
    offline_params.pop('radius_bound', None)
    offline_params.pop('num_train_samples', None)
    offline_params.pop('num_test_samples', None)
    offline_params['train_data'] = train_path
    offline_params['test_data'] = test_path
    return offline_params


def _overwrite_params_with_cmd_arguments(params, args):
    # This function overwrites parameters in the given dictionary
    # with the correspondent command line arguments.
    if args.name is not None:
        params['experiment_id'] = args.name[0]
    if args.ntrain is not None:
        params['dataset']['num_train_samples'] = args.ntrain[0]
    if args.ntest is not None:
        params['dataset']['num_test_samples'] = args.ntest[0]
    if args.env is not None:
        env_params = _read_params(DEFAULT_ENVIRONMENTS_PATH + args.env[0] + '.yaml')
        params['environment'] = env_params['environment']
    if args.env_spec is not None:
        for spec in args.env_spec:
            key, value = spec.split(':')
            params['environment'][key] = ast.literal_eval(value)


if __name__ == '__main__':
    DATASETS_ROOT = os.path.join(os.path.dirname(__file__), 'datasets')
    DEFAULT_PARAMS_FILE = os.path.join(
        os.path.dirname(__file__), 'experiment_params/default_online.yaml')
    DEFAULT_ENVIRONMENTS_PATH = os.path.join(
        os.path.dirname(__file__), 'experiment_params/', 'default_environments/'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', action='store', nargs=1, type=str, required=False,
        help='YAML file from which to read the dataset parameters. If not specified,'
             'experiment_params/default_online.yaml will be used.')
    parser.add_argument('--name', action='store', nargs=1, required=False,
                        help='Use this name for the dataset instead of experiment_name in the '
                             'yaml file.')
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
        help='The default environment specifications to use. Can be \'pendulum\', \'spring\', '
             '\'two_bodies\', \'three_bodies\', \'chaotic_pendulum\'. If this argument is '
             'specified, a default environment section will be loaded from the correspondent yaml '
             'file in experiment_params/default_environments/'
    )
    parser.add_argument(
        '--env-spec', action='store', nargs='+', required=False,
        help='Parameters of the environment in the form param_name:param_value, '
             'e.g. --env-spec g:1.0 mass:0.5. If this argument is specified, the given '
             'parameters will be used instead of those in the yaml file. IMPORTANT: lists must be '
             'enclosed in double quotes, i.e. mass:"[0.5, 0.5]".'
    )
    args = parser.parse_args()

    # Read yaml file with parameters definition
    parameter_file = args.params[0] if args.params is not None else DEFAULT_PARAMS_FILE
    dataset_params = _read_params(parameter_file)

    # Overwrite dictionary from command line args to ensure they will be used
    _overwrite_params_with_cmd_arguments(dataset_params, args)

    # Extract environment parameters
    EXP_NAME = dataset_params['experiment_id']
    N_TRAIN_SAMPLES = dataset_params['dataset']['num_train_samples']
    N_TEST_SAMPLES = dataset_params['dataset']['num_test_samples']
    IMG_SIZE = dataset_params['dataset']['img_size']
    RADIUS_BOUND = dataset_params['dataset']['radius_bound']
    NOISE_LEVEL = dataset_params['dataset']['rollout']['noise_level']
    N_FRAMES = dataset_params['dataset']['rollout']['seq_length']
    DELTA_TIME = dataset_params['dataset']['rollout']['delta_time']
    N_CHANNELS = dataset_params['dataset']['rollout']['n_channels']

    # Get dataset output path
    DATASETS_ROOT = os.path.join(DATASETS_ROOT, EXP_NAME)

    # Get the environment object from dictionary parameters
    environment = environment_factory.EnvFactory.get_environment(**dataset_params['environment'])

    # Ask user confirmation
    print(f'The dataset will be generated with the following parameters:')
    print(f'experiment_id: {EXP_NAME}')
    print(f'dataset: {dataset_params["dataset"]}')
    print(f'environment: {dataset_params["environment"]}')
    print('\nProceed? (y/n):')
    if input() != 'y':
        print('Aborting')
        exit()

    # Generate train samples
    train_path = generate_and_save(
        root_path=DATASETS_ROOT, environment=environment,
        n_samples=N_TRAIN_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME, img_size=IMG_SIZE,
        radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL, color=N_CHANNELS == 3,
        start_seed=0, train=True
    )

    # Generate test samples
    test_path = None
    if N_TEST_SAMPLES > 0:
        test_path = generate_and_save(
            root_path=DATASETS_ROOT, environment=environment,
            n_samples=N_TEST_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME,
            img_size=IMG_SIZE, radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL,
            color=N_CHANNELS == 3, start_seed=N_TRAIN_SAMPLES, train=False
        )

    # Convert parameters to offline train parameters and write them in the dataset
    out_params = _prepare_out_params(dataset_params, train_path, test_path)
    yaml_content = yaml.dump(out_params, default_flow_style=True)
    param_out_path = os.path.join(DATASETS_ROOT, 'parameters.yaml')
    with open(param_out_path, 'x') as f:
        f.write(yaml_content)
        f.close()

    print(f'A parameter file ready to be trained on was generated at {param_out_path}')
