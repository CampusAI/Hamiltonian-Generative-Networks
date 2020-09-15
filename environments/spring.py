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
            p (float, optional): Generalized momentum. Defaults to None.
            q ([type], optional): Generalized position. Defaults to None.
        """
        self.mass = mass
        self.elastic_cst = elastic_cst
        self.set(p, q)

    def set(self, p, q):
        """Sets initial conditions for spring

        Args:
            p (float): Generalized momentum
            q (float): Generalized position
        """
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

        self.q += dt*(self.p/(self.mass))
        self.p += dt*-self.elastic_cst*self.q

    def draw(self):
        """Caption from the actual spring state

        Returns:
            Img (np.ndarray): Caption of current state
        """
        img = Image.new('L', (32, 32))
        draw = ImageDraw.Draw(img)

        r = self.mass
        x = self.q + 32/2
        y = 32/2

        draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

        return np.array(img)
