from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
import numpy as np


class Environment(ABC):

    @abstractmethod
    def set(self, p, q):
        """Sets initial conditions for physical system

        Args:
            p (float): generalized momentum
            q (float): generalized position

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt=0.01):
        """Performs a step in the environment simulator

        Args:
            dt (float, optional): Time step run for the integration. Defaults to 0.01.

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self):
        """Returns Array of the environment state

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    def generate_data(self, total_seconds, fps, dt=0.01):
        """Generates dataset for current environemnt

        Args:
            total_seconds (int): Total duration of video (in seconds)
            fps (int): Frame rate of generated data (frames per second)
            dt (float, optional): Time step run for the integration. Defaults to 0.01.

        Returns:
            (dict): Contains input data and corresponding ground truth phase states
        """

        time_evol = 1./fps
        total_images = total_seconds*fps
        image_list = [self.draw()]
        phase_state_list = [np.array([self.q, self.p])]

        for _ in range(total_images):
            current_time = 0
            while(current_time < time_evol):
                self.step(dt)
                current_time += dt
            image_list.append(self.draw())
            phase_state_list.append(np.array([self.q, self.p]))
        return {'input': np.array(image_list), 'groundtruth': np.array(phase_state_list)}
