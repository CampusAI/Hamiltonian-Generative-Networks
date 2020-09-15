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

    def draw(self):
        img = Image.new('L', (32, 32))
        draw = ImageDraw.Draw(img)

        r = self.mass
        x = np.sin(self.q) * self.length + 32 / 2
        y = np.cos(self.q) * self.length

        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

        return np.array(img)


if __name__ == "__main__":
    env = Pendulum(mass=1, length=20)
    env.set(15, 0.5)
    img = env.draw()
    import cv2
    cv2.imshow("img", img)
    cv2.waitKey(0)