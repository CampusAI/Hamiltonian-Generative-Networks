import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pendulum import Pendulum


class EnvironmentSampler(Dataset):
    """Dataset for rollout sampling
    Given an environment and sampling conditions, the dataset samples rollouts as pytorch tensors.
    """

    def __init__(self,
                 environment,
                 dataset_len,
                 number_of_frames,
                 delta_time,
                 number_of_rollouts,
                 img_size,
                 noise_std,
                 radius_bound,
                 world_size,
                 seed,
                 data_mean=.5,
                 data_std=.5):
        """Constructor method

        Args:
            environment (Environment): Instance belonging to Environment abstract base class.
            dataset_len (int): Length of the dataset.
            number_of_frames (int): Total duration of video (in frames).
            delta_time (float): Frame interval of generated data (in seconds).
            number_of_rollouts (int): Number of rollouts to generate.
            img_size (int): Size of the frames (in pixels).
            color (bool): Whether to have colored or grayscale frames.
            noise_std (float): Standard deviation of the gaussian noise source, no noise for noise_std=0.
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
                Init phase states will be sampled from a circle (q, p) of radius
                r ~ U(radius_bound[0], radius_bound[1]) https://arxiv.org/pdf/1909.13789.pdf (Sec. 4)
            world_size (float) Spatial extent of the window where the rendering is taking place (in meters).
            seed (int): Seed for reproducibility.
            data_mean (float): Data mean for standaritzation. Defaults to 0.5.
            data_std (float): Data std for standaritzation. Defaults to 0.5.
        """
        self.environment = environment
        self.dataset_len = dataset_len
        self.data_mean = data_mean
        self.data_std = data_std
        self.number_of_frames = number_of_frames
        self.delta_time = delta_time
        self.number_of_rollouts = number_of_rollouts
        self.img_size = img_size
        self.noise_std = noise_std
        self.radius_bound = radius_bound
        self.world_size = world_size
        self.seed = seed

    def __len__(self):
        """Get dataset length

        Returns:
            length (int): Length of the dataset.
        """
        return self.dataset_len

    def __getitem__(self, i):
        """Iterator for rollout sampling.
        Samples a rollout and converts it to a Pytorch tensor.

        Args:
            i (int): Index of the dataset sample (ignored since we sample random data).
        Returns:
            (Torch.tensor): Tensor of shape (Batch, Nframes, Channels, Height, Width).
                Contains standarized sampled rollouts        
        """
        rolls = self.environment.sample_random_rollouts(
            number_of_frames=self.number_of_frames,
            delta_time=self.delta_time,
            number_of_rollouts=self.number_of_rollouts,
            img_size=self.img_size,
            noise_std=self.noise_std,
            radius_bound=self.radius_bound,
            world_size=self.world_size,
            seed=self.seed)
        # standarization
        rolls = (rolls - self.data_mean) / self.data_std
        return rolls.transpose((0, 1, 4, 2, 3))


class EnvironmentLoader(Dataset):
    def __init__(self, root_dir, data_mean=1., data_std=1.):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        rolls = np.load(os.path.join(
            self.root_dir, self.file_list[i]))['arr_0']
        rolls = (rolls - self.data_mean) / self.data_std
        return rolls


# Sample code for DataLoader call
if __name__ == "__main__":
    import time
    pd = Pendulum(mass=.5, length=1, g=3)
    trainDS = EnvironmentSampler(environment=pd,
                                 dataset_len=100,
                                 data_mean=.5,
                                 data_std=.5,
                                 number_of_frames=100,
                                 delta_time=.1,
                                 number_of_rollouts=4,
                                 img_size=64,
                                 noise_std=0.,
                                 radius_bound=(1.3, 2.3),
                                 world_size=1.5,
                                 seed=23)
    # Dataloader instance test, batch_mode disabled
    train = torch.utils.data.DataLoader(trainDS,
                                        shuffle=False,
                                        batch_size=None)
    start = time.time()
    sample = next(iter(train))
    end = time.time()

    print(sample.size(), "Sampled in " + str(end - start) + " s")

    # Dataloader instance test, batch_mode enabled
    train = torch.utils.data.DataLoader(trainDS,
                                        shuffle=False,
                                        batch_size=4,
                                        num_workers=1)
    start = time.time()
    sample = next(iter(train))
    end = time.time()

    print(sample.size(), "Sampled in " + str(end - start) + " s")

    trainDS = EnvironmentLoader('../datasets/pendulum_data/train')

    train = torch.utils.data.DataLoader(trainDS,
                                        shuffle=False,
                                        batch_size=10,
                                        num_workers=4)
    start = time.time()
    sample = next(iter(train))
    end = time.time()
    print(sample.size(), "Sampled in " + str(end - start) + " s")
