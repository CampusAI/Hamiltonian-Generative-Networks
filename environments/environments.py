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

    def step(self, dt=0.01):
        """Performs a step in the environment simulator

        Args:
            dt (float, optional): [description]. Defaults to 0.01.
        """
        raise NotImplementedError

    def draw(self):
        """Returns Caption of the environment state
        """
        raise NotImplementedError

    def generateData(self, total_seconds, fps):
        """Generates dataset for current environemnt

        Args:
            total_seconds (int): Total duration of video (in seconds)
            fps (int): Frame rate of generated data (frames per second)
        
        Returns dict containing input data and corresponding ground truth phase states
        """

        time_evol = 1./fps
        total_images = total_seconds*fps
        dt = 0.01
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


class Pendulum(Environment):

    """Pendulum System

    Equations of movement are:

        theta'' = -(g/l)*sin(theta)

    """

    def __init__(self, mass, length, p=None, q=None):
        self.mass = mass
        self.length = length
        self.set(p, q)

    def set(self, p, q):
        self.p = p
        self.q = q

    def step(self, dt=0.01):

        self.q += dt*(self.p/(self.mass*self.length*self.length))
        self.p += dt*-9.81*self.mass*self.length*np.sin(self.q)

    def draw(self):
        img = Image.new('L', (32, 32))
        draw = ImageDraw.Draw(img)

        r = self.mass
        x = np.sin(self.q)*self.length + 32/2
        y = np.cos(self.q)*self.length

        draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

        return np.array(img)
