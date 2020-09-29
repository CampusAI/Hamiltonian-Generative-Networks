from abc import ABC, abstractmethod
import numpy as np
import os


class Environment(ABC):
    @abstractmethod
    def set(self, p, q):
        """Sets initial conditions for physical system

        Args:
            p ([float]): generalized momentum in n-d space
            q ([float]): generalized position in n-d space

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

    def generate_data(self, total_seconds, fps, save_dir= None, dt=0.01):
        """Generates dataset for current environemnt

        Args:
            total_seconds (int): Total duration of video (in seconds)
            fps (int): Frame rate of generated data (frames per second)
            save_dir (string, optional): If not None then save the dataset in directory. Defaults to None.
            dt (float, optional): Time step run for the integration. Defaults to 0.01.

        Returns:
            (dict): Contains frames and corresponding phase states
        """

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
