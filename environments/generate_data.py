import os

import numpy as np
from tqdm import tqdm

from pendulum import Pendulum
from spring import Spring

# PARAMETERS

dataset_root = '../datasets'
dataset_name = "pendulum_data"
num_train_samples = 1000
num_test_samples = 100
number_of_frames = 20
delta_time = 0.125
img_size = 32
noise_std = 0.
radius_bound = (.5, 1.5)
color = False
world_size = 1.5
environment = Pendulum(mass=.5, length=1, g=3)

############

dataset_path_train = os.path.join(dataset_root, dataset_name, 'train')
os.makedirs(dataset_path_train, exist_ok=True)
print("Generating train dataset...")
for i in tqdm(range(num_train_samples)):
    rolls = environment.sample_random_rollouts(number_of_frames=number_of_frames,
                                               delta_time=delta_time,
                                               number_of_rollouts=1,
                                               img_size=img_size,
                                               noise_std=noise_std,
                                               radius_bound=radius_bound,
                                               world_size=world_size,
                                               color=color,
                                               seed=i)[0]
    filename = "{0:05d}".format(i)
    np.savez(os.path.join(dataset_path_train, filename), rolls)

dataset_path_test = os.path.join(dataset_root, dataset_name, 'test')
os.makedirs(dataset_path_test, exist_ok=True)
print("Generating test dataset...")
for i in tqdm(range(num_test_samples)):
    rolls = environment.sample_random_rollouts(number_of_frames=number_of_frames,
                                               delta_time=delta_time,
                                               number_of_rollouts=1,
                                               img_size=img_size,
                                               noise_std=noise_std,
                                               radius_bound=radius_bound,
                                               world_size=world_size,
                                               color=color,
                                               seed=num_test_samples + i)[0]
    filename = "{0:05d}".format(i)
    np.savez(os.path.join(dataset_path_test, filename), rolls)
