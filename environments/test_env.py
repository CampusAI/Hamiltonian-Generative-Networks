from .environments import Environment
from PIL import Image, ImageDraw
import numpy as np


class TestEnv(Environment):
    """Testing environment
    """

    def __init__(self, q=None, p=None):
        self.set(p, q)

    def set(self, q, p):
        """Sets initial conditions for TestEnv

        Args:
            q ([float]): Generalized position in 1-D space
            p ([float]): Generalized momentum in 1-D space
        
        Raises:
            ValueError: If p and q are not in 1-D space
        """
        self.p = p
        self.q = q

    def step(self, dt=1.0):
        self.p += dt

    def draw(self):
        return self.p
