import argparse
import sys
import os
import yaml

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_factory import EnvFactory
from pendulum import Pendulum
from spring import Spring


def generate_and_save(datasets_root, dataset_name, environment, n_samples, n_frames,
                      delta_time, img_size, radius_bound, noise_level, color, train=True):
    train_path = os.path.join(datasets_root, dataset_name, 'train' if train else 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    for i in tqdm(range(n_samples)):
        rolls = environment.sample_random_rollouts(
            number_of_frames=n_frames,
            delta_time=delta_time,
            number_of_rollouts=1,
            img_size=img_size,
            noise_level=noise_level,
            radius_bound=radius_bound,
            color=color,
            seed=i
        )[0]
        filename = "{0:05d}".format(i)
        np.savez(os.path.join(train_path, filename), rolls)


def _read_params(params_file):
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


if __name__ == '__main__':
    DATASETS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    DEFAULT_PARAMS_FILE = os.path.join(
        os.path.dirname(__file__), '..', 'experiment_params/default_online.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params', action='store', nargs=1, type=str, required=False,
        help='YAML file from which to read the dataset parameters. If not specified,'
             'experiment_params/default_online.yaml will be used.')
    parser.add_argument('--name', action='store', nargs=1, required=False,
                        help='Use this name for the dataset instead of experiment_name in the '
                             'yaml file.')
    args = parser.parse_args()

    parameter_file = args.params[0] if args.params is not None else DEFAULT_PARAMS_FILE
    online_params = _read_params(parameter_file)

    try:
        EXP_NAME = online_params['experiment_id'] if args.name is None else args.name[0]
        N_TRAIN_SAMPLES = online_params['dataset']['num_train_samples']
        N_TEST_SAMPLES = online_params['dataset']['num_test_samples']
        IMG_SIZE = online_params['dataset']['img_size']
        RADIUS_BOUND = online_params['dataset']['radius_bound']
        NOISE_LEVEL = online_params['dataset']['rollout']['noise_level']
        N_FRAMES = online_params['dataset']['rollout']['seq_length']
        DELTA_TIME = online_params['dataset']['rollout']['delta_time']
        N_CHANNELS = online_params['dataset']['rollout']['n_channels']
    except KeyError:
        raise KeyError(f'The given parameter file {parameter_file} does not fully specify the ' +
                       'required parameters.')

    environment = EnvFactory.get_environment(**online_params['environment'])

    # Generate train samples
    generate_and_save(datasets_root=DATASETS_ROOT, dataset_name=EXP_NAME, environment=environment,
                      n_samples=N_TRAIN_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME,
                      img_size=IMG_SIZE, radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL,
                      color=N_CHANNELS == 3, train=True)

    # Generate test samples
    if N_TEST_SAMPLES > 0:
        generate_and_save(datasets_root=DATASETS_ROOT, dataset_name=EXP_NAME,
                          environment=environment,
                          n_samples=N_TRAIN_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME,
                          img_size=IMG_SIZE, radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL,
                          color=N_CHANNELS == 3, train=False)
