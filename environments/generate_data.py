import argparse
import copy
import shutil
import sys
import os
import yaml

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment_factory import EnvFactory
from pendulum import Pendulum
from spring import Spring


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


def _online_params_to_offline_params(params, train_path, test_path):
    offline_params = copy.deepcopy(params)
    offline_params.pop('img_size', None)
    offline_params.pop('radius_bound', None)
    offline_params.pop('num_train_samples', None)
    offline_params.pop('num_test_samples', None)
    offline_params['train_data'] = train_path
    offline_params['test_data'] = test_path
    return offline_params


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
    parser.add_argument(
        '--ntrain', action='store', nargs=1, required=False, type=int,
        help='Number of training sample to generate.'
    )
    parser.add_argument(
        '--ntest', action='store', nargs=1, required=False, type=int,
        help='Number of test samples to generate.'
    )
    args = parser.parse_args()

    parameter_file = args.params[0] if args.params is not None else DEFAULT_PARAMS_FILE
    online_params = _read_params(parameter_file)

    # Overwrite dictionary with command line args
    if args.name is not None:
        online_params['experiment_id'] = args.name[0]
    if args.ntrain is not None:
        online_params['dataset']['num_train_samples'] = args.ntrain[0]
    if args.ntest is not None:
        online_params['dataset']['num_test_samples'] = args.ntest[0]

    try:
        EXP_NAME = online_params['experiment_id']
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

    DATASETS_ROOT = os.path.join(DATASETS_ROOT, EXP_NAME)

    environment = EnvFactory.get_environment(**online_params['environment'])

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
            n_samples=N_TRAIN_SAMPLES, n_frames=N_FRAMES, delta_time=DELTA_TIME,
            img_size=IMG_SIZE, radius_bound=RADIUS_BOUND, noise_level=NOISE_LEVEL,
            color=N_CHANNELS == 3, start_seed=N_TRAIN_SAMPLES, train=False
        )

    offline_params = _online_params_to_offline_params(online_params, train_path, test_path)
    yaml_content = yaml.dump(offline_params, default_flow_style=True)
    with open(os.path.join(DATASETS_ROOT, 'parameters.yaml'), 'x') as f:
        f.write(yaml_content)
        f.close()
