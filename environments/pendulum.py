from .environments import Environment
from PIL import Image, ImageDraw
import numpy as np
from skimage.draw import circle


class Pendulum(Environment):

    """Pendulum System

    Equations of movement are:

        theta'' = -(g/l)*sin(theta)

    """

    def __init__(self, mass, length, g, p=None, q=None):
        """Contructor for pendulum system

        Args:
            mass (float): Pendulum mass
            length (float): Pendulum length
            g (float): Gravity of the environment
            p ([float], optional): Generalized momentum in 1-D space: Angular momentum. Defaults to None
            q ([float], optional): Generalized position in 1-D space: Phase. Defaults to None
        """
        self.mass = mass
        self.length = length
        self.g = g
        self.set(p, q)
        super().__init__()

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
        self.p[0] += dt*-self.g*self.mass*self.length*np.sin(self.q[0])

    def dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        return [(states[1]/(self.mass*self.length*self.length)), -self.g*self.mass*self.length*np.sin(states[0])]

    def draw(self, res=32, color=True):
        """Returns array of the environment evolution

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self.rollout[0,:]
        length = len(q)
        if color:
            vid = np.zeros((length, res, res, 3), dtype='float')
        else:
            vid = np.zeros((length, res, res, 1), dtype='float')
        SIZE = 1.5
        grid = np.arange(0, 1, 1. / res) *2*SIZE  - SIZE
        [I, J] = np.meshgrid(grid, grid)
        for t in range(length):
            if color:
                vid[t, :, :, 0] += np.exp(-(((I - np.sin(q[t])) ** 2 + (J - np.cos(q[t])) ** 2) /(self.mass ** 2)) ** 4)
                vid[t, :, :, 1] += np.exp(-(((I - np.sin(q[t])) ** 2 + (J - np.cos(q[t])) ** 2) /(self.mass ** 2)) ** 4)
            else:
                vid[t, :, :, 0] += np.exp(-(((I - np.sin(q[t])) ** 2 + (J - np.cos(q[t])) ** 2) /(self.mass ** 2)) ** 4)
            vid[t][vid[t] > 1] = 1

        return vid

    def sample_init_conditions(self, radius):
        """Samples random initial conditions for the environment

        Args:
            radius (float): Radius of the sampling process
        """
        states = np.random.rand(2)*2.-1
        states /= np.sqrt((states**2).sum())*radius
        self.set([states[0]], [states[1]])
