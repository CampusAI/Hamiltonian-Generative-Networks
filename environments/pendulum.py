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
        return [(states[1]/(self.mass*self.length*self.length)), -self.g*self.mass*self.length*np.sin(states[0])]

    def draw(self, size=32):
        """Caption from the actual pendulum state

        Returns:
            Img (np.ndarray): Caption of current state
        """
        q = self.rollout[0,:]
        rollout_imgs = []
        for i in range(len(q)):
            img = np.zeros((size, size, 1), np.uint8)

            r = self.mass
            y = int(np.sin(q[i])*self.length + size/2)
            x = int(np.cos(q[i])*self.length)
            rr, cc = circle(x,y,r, shape=img.shape)
            img[rr, cc] = 255

            rollout_imgs.append(np.array(img))
        return np.array(rollout_imgs)

    def sample_init_conditions(self):
        self.set([np.random.rand()*2.-1], [np.random.rand()*2.-1])