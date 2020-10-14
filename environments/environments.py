from abc import ABC, abstractmethod
import os

import numpy as np
from scipy.integrate import solve_ivp


class Environment(ABC):
    def __init__(self, q=None, p=None):
        """Instantiate new environment with the provided position and momentum

        Args:
            q ([float], optional): generalized position in n-d space
            p ([float], optional): generalized momentum in n-d space
        """
        self._rollout = None
        self.q = None
        self.p = None
        self.set(q=q, p=p)

class Environment(ABC):
    @abstractmethod
    def set(self, q, p):
        """Sets initial conditions for physical system

        Args:
            q ([float]): generalized position in n-d space
            p ([float]): generalized momentum in n-d space

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def _draw(self):
        """Returns array of the environment evolution

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    def _sample_init_conditions(self, radius):
        """Samples random initial conditions for the environment

        Args:
            radius (float): Radius of the sampling process

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

<<<<<<< HEAD
    def generate_data(self, total_seconds, fps, save_dir= None, dt=0.01):
        """Generates dataset for current environment
=======
    def _evolution(self, total_time=10, delta_time=0.1):
        """Performs rollout of the physical system given some initial conditions.
        Sets rollout phase states to self.rollout

        Args:
            total_time (float): Total duration of the rollout (in seconds)
            delta_time (float): Sample interval in the rollout (in seconds)

        Raises:
            AssertError: If p or q are None
        """
        assert self.q != None
        assert self.p != None

        t_eval = np.linspace(0, total_time,
                             round(total_time / delta_time) + 1)[:-1]
        t_span = [0, total_time]
        y0 = np.array([np.array(self.q), np.array(self.p)]).reshape(-1)
        self._rollout = solve_ivp(self._dynamics, t_span, y0, t_eval=t_eval).y

    def sample_random_rollouts(self,
                               number_of_frames=100,
                               delta_time=0.1,
                               number_of_rollouts=16,
                               img_size=32,
                               color=True,
                               noise_std=0.,
                               base_radius=1.3,
                               seed=None):
        """Samples random rollouts for a given environment
>>>>>>> master

        Args:
            number_of_frames (int): Total duration of video (in frames)
            delta_time (float): Frame interval of generated data (in seconds)
            number_of_rollouts (int): Number of rollouts to generate.
            img_size (int): Size of the frames (in pixels)
            color (bool): Whether to have colored or grayscale frames.
            noise_std (float): If noisy_data=true, standard deviation of the gaussian noise source.
            base_radius (float): Init phase states will be sampled from a circle (q, p) of radius 
                r ~ U(base_radius, base_radius + 1) https://arxiv.org/pdf/1909.13789.pdf (Sec. 4)
            seed (int): Seed for reproducibility

        Returns:
            (ndarray): Array of shape (Batch, Nframes, Height, Width, Channels).
                Contains sampled rollouts
        """
        if seed is not None:
            np.random.seed(seed)
        total_time = number_of_frames * delta_time
        batch_sample = []
        for i in range(number_of_rollouts):
            radius = np.random.rand() + base_radius
            self._sample_init_conditions(radius)
            self._evolution(total_time, delta_time)
            if noise_std > 0.:
                self._rollout += np.random.randn(
                    *self._rollout.shape) * noise_std
            batch_sample.append(self._draw(img_size, color))

<<<<<<< HEAD
        time_evol = 1./fps
        total_images = total_seconds*fps
        image_list = [self.draw()]
        phase_state_list = [np.array([np.array(self.q), np.array(self.p)])]

        for _ in range(total_images):
            current_time = 0
            while(current_time < time_evol):
                self.step(dt)
                current_time += dt
            image_list.append(self.draw())
            phase_state_list.append(np.array([np.array(self.q), np.array(self.p)]))

        dataset = {'frames': np.array(image_list), 'states': np.array(phase_state_list)}
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'dataset'), dataset)
        return dataset

    def sample_random_rollout(self, seed=0, number_of_frames=6, delta_time=1, number_of_rollouts=1):
        return torch.tensor([[[1], [2], [3], [4]]])
=======
        return np.array(batch_sample)
>>>>>>> master
