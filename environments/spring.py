from .environments import Environment
from PIL import Image, ImageDraw
import numpy as np


class Spring(Environment):

    """Spring System

    Equations of movement are:

        x'' = -(k/m)*x

    """

    def __init__(self, mass, elastic_cst, p=None, q=None):
        """Contructor for spring system

        Args:
            mass (float): Spring mass
            elastic_cst (float): Spring elastic constant
            p ([float], optional): Generalized momentum in 1-D space: Linear momentum. Defaults to None
            q ([float], optional): Generalized position in 1-D space: Position. Defaults to None
        """
        self.mass = mass
        self.elastic_cst = elastic_cst
        self.set(p, q)

    def set(self, p, q):
        """Sets initial conditions for pendulum

        Args:
            p ([float]): Generalized momentum in 1-D space: Linear momentum
            q ([float]): Generalized position in 1-D space: Position
        
        Raises:
            ValueError: If p and q are not in 1-D space
        """
        if len(p) != 1 or len(q) != 1:
            raise ValueError("p and q must be in 1-D space: Linear momentum and Position.")
        self.p = p
        self.q = q

    def step(self, dt=0.01):
        """Performs a step in the spring system

        Args:
            dt (float, optional): Time step run for the integration. Defaults to 0.01.

        Raises:
            TypeError: If p or q are None
        """

        assert type(self.q) != None
        assert type(self.p) != None

        self.q[0] += dt*(self.p[0]/(self.mass))
        self.p[0] += dt*-self.elastic_cst*self.q[0]

    def draw(self):
        """Caption from the actual spring state

        Returns:
            Img (np.ndarray): Caption of current state
        """
        img = Image.new('L', (32, 32))
        draw = ImageDraw.Draw(img)

        r = self.mass
        x = self.q[0] + 32/2
        y = 32/2

        draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

        return np.array(img)
