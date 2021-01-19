from abc import ABC, abstractmethod
import os

import cv2
from matplotlib import pyplot as plt, animation
import numpy as np
from scipy.integrate import solve_ivp


class Environment(ABC):
    def __init__(self, q=None, p=None):
        """Instantiate new environment with the provided position and momentum

        Args:
            q ([float], optional): generalized position in n-d space
            p ([float], optional): generalized momentum in n-d space
        """
        self._default_background_color = [81./255, 88./255, 93./255]
        self._default_ball_colors = [
            (173./255, 146./255, 0.), (173./255, 0., 0.), (0., 146./255, 0.)]
        self._rollout = None
        self.q = None
        self.p = None
        self.set(q=q, p=p)

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
    def _draw(self, img_size, color):
        """Returns array of the environment evolution.

        Args:
            img_size (int): Size of the frames (in pixels).
            color (bool): Whether to have colored or grayscale frames.

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def get_world_size(self):
        """Returns the world size for the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_noise_std(self):
        """Returns the maximum noise standard deviation that maintains a stable environment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_radius_bounds(self):
        """Returns a tuple (min, max) with the default radius bounds for the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def _sample_init_conditions(self, radius_bound):
        """Samples random initial conditions for the environment

        Args:
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    def _world_to_pixels(self, x, y, res):
        """Maps coordinates from world space to pixel space

        Args:
            x (float): x coordinate of the world space.
            y (float): y coordinate of the world space.
            res (int): Image resolution in pixels (images are square).

        Returns:
            (int, int): Tuple of coordinates in pixel space.
        """
        pix_x = int(res*(x + self.get_world_size())/(2*self.get_world_size()))
        pix_y = int(res*(y + self.get_world_size())/(2*self.get_world_size()))

        return (pix_x, pix_y)

    def _evolution(self, total_time=10, delta_time=0.1):
        """Performs rollout of the physical system given some initial conditions.
        Sets rollout phase states to self.rollout

        Args:
            total_time (float): Total duration of the rollout (in seconds)
            delta_time (float): Sample interval in the rollout (in seconds)

        Raises:
            AssertError: If p or q are None
        """
        if isinstance(self.q, np.ndarray):
            assert self.q.all() != None
            assert self.p.all() != None
        else:
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
                               noise_level=0.1,
                               radius_bound=(1.3, 2.3),
                               seed=None):
        """Samples random rollouts for a given environment

        Args:
            number_of_frames (int): Total duration of video (in frames).
            delta_time (float): Frame interval of generated data (in seconds).
            number_of_rollouts (int): Number of rollouts to generate.
            img_size (int): Size of the frames (in pixels).
            color (bool): Whether to have colored or grayscale frames.
            noise_level (float): Level of noise, in [0, 1]. 0 means no noise, 1 means max noise.
                Maximum noise is defined in each environment.
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
                Init phase states will be sampled from a circle (q, p) of radius
                r ~ U(radius_bound[0], radius_bound[1]) https://arxiv.org/pdf/1909.13789.pdf (Sec. 4)
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.
            seed (int): Seed for reproducibility.
        Raises:
            AssertError: If radius_bound[0] > radius_bound[1]
        Returns:
            (ndarray): Array of shape (Batch, Nframes, Height, Width, Channels).
                Contains sampled rollouts
        """
        if radius_bound == 'auto':
            radius_bound = self.get_default_radius_bounds()
        radius_lb, radius_ub = radius_bound
        assert radius_lb <= radius_ub
        if seed is not None:
            np.random.seed(seed)
        total_time = number_of_frames * delta_time
        batch_sample = []
        for i in range(number_of_rollouts):
            self._sample_init_conditions(radius_bound)
            self._evolution(total_time, delta_time)
            if noise_level > 0.:
                self._rollout += np.random.randn(
                    *self._rollout.shape) * noise_level * self.get_max_noise_std()
            batch_sample.append(self._draw(img_size, color))

        return np.array(batch_sample)


def visualize_rollout(rollout, interval=50, show_step=False):
    """Visualization for a single sample rollout of a physical system.

    Args:
        rollout (numpy.ndarray): Numpy array containing the sequence of images. It's shape must be
            (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    fig = plt.figure()
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape))
            cv2.putText(
                black_img, str(i), (0, 30), fontScale=0.22, color=(255, 255, 255), thickness=1,
                fontFace=cv2.LINE_AA)
            res_img = (im + black_img / 255.) / 2
        else:
            res_img = im
        img.append([plt.imshow(res_img, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=100)
    plt.show()
