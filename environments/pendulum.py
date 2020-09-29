from .environments import Environment
from PIL import Image, ImageDraw
import numpy as np


class Pendulum(Environment):

    """Pendulum System

    Equations of movement are:

        theta'' = -(g/l)*sin(theta)

    """

    def __init__(self, mass, length, p=None, q=None):
        """Contructor for pendulum system

        Args:
            mass (float): Pendulum mass
            length (float): Pendulum length
            p ([float], optional): Generalized momentum in 1-D space: Angular momentum. Defaults to None
            q ([float], optional): Generalized position in 1-D space: Phase. Defaults to None
        """
        self.mass = mass
        self.length = length
        self.set(p, q)

    def set(self, p, q):
        """Sets initial conditions for pendulum

        Args:
            p ([float]): Generalized momentum in 1-D space: Angular momentum
            q ([float]): Generalized position in 1-D space: Phase
        
        Raises:
            ValueError: If p and q are not in 1-D space
        """
        if len(p) != 1 or len(q) != 1:
            raise ValueError("p and q must be in 1-D space: Angular momentum and Phase.")
        self.p = p
        self.q = q

    def step(self, dt=0.01):
        """Performs a step in the pendulum system

        Args:
            dt (float, optional): Time step run for the integration. Defaults to 0.01.

        Raises:
            TypeError: If p or q are None
        """

        assert type(self.q) != None
        assert type(self.p) != None

        self.q[0] += dt*(self.p[0]/(self.mass*self.length*self.length))
        self.p[0] += dt*-9.81*self.mass*self.length*np.sin(self.q[0])

    def draw(self):
        """Caption from the actual pendulum state

        Returns:
            Img (np.ndarray): Caption of current state
        """
        img = Image.new('L', (32, 32))
        draw = ImageDraw.Draw(img)

        r = self.mass
        x = np.sin(self.q[0])*self.length + 32/2
        y = np.cos(self.q[0])*self.length

        draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

        return np.array(img)
