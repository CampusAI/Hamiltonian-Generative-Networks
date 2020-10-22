import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import conversions


class EnvironmentSampler(Dataset):
    """Dataset for rollout sampling.

    Given an environment and sampling conditions, the dataset samples rollouts as pytorch tensors.
    """

    def __init__(self,
                 environment,
                 dataset_len,
                 number_of_frames,
                 delta_time,
                 number_of_rollouts,
                 img_size,
                 color,
                 noise_level,
                 radius_bound,
                 seed,
                 dtype=torch.float):
        """Instantiate the EnvironmentSampler.

        Args:
            environment (Environment): Instance belonging to Environment abstract base class.
            dataset_len (int): Length of the dataset.
            number_of_frames (int): Total duration of video (in frames).
            delta_time (float): Frame interval of generated data (in seconds).
            number_of_rollouts (int): Number of rollouts to generate.
            img_size (int): Size of the frames (in pixels).
            color (bool): Whether to have colored or grayscale frames.
            noise_level (float): Value in [0, 1] to tune the noise added to the environment
                trajectory.
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
                Init phase states will be sampled from a circle (q, p) of radius
                r ~ U(radius_bound[0], radius_bound[1]) https://arxiv.org/pdf/1909.13789.pdf (Sec 4)
                Optionally, it can be a string 'auto'. In that case, the value returned by
                environment.get_default_radius_bounds() will be returned.
            seed (int): Seed for reproducibility.
            dtype (torch.type): Type of the sampled tensors.
        """
        self.environment = environment
        self.dataset_len = dataset_len
        self.number_of_frames = number_of_frames
        self.delta_time = delta_time
        self.number_of_rollouts = number_of_rollouts
        self.img_size = img_size
        self.color = color
        self.noise_level = noise_level
        self.radius_bound = radius_bound
        self.seed = seed
        self.dtype = dtype

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
            (Torch.tensor): Tensor of shape (batch_len, seq_len, channels, height, width) with the
                sampled rollouts.
        """
        rolls = self.environment.sample_random_rollouts(
            number_of_frames=self.number_of_frames,
            delta_time=self.delta_time,
            number_of_rollouts=self.number_of_rollouts,
            img_size=self.img_size,
            color=self.color,
            noise_level=self.noise_level,
            radius_bound=self.radius_bound,
            seed=self.seed)
        rolls = torch.from_numpy(rolls).type(self.dtype)
        return conversions.to_channels_first(rolls)


class EnvironmentLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        rolls = np.load(os.path.join(
            self.root_dir, self.file_list[i]))['arr_0']
        return rolls.transpose((0, 3, 1, 2))


# Sample code for DataLoader call
if __name__ == "__main__":
    import time
    from .pendulum import Pendulum

    pd = Pendulum(mass=.5, length=1, g=3)
    trainDS = EnvironmentSampler(environment=pd,
                                 dataset_len=100,
                                 number_of_frames=100,
                                 delta_time=.1,
                                 number_of_rollouts=4,
                                 img_size=64,
                                 noise_level=0.,
                                 radius_bound=(1.3, 2.3),
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
