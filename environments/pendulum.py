from matplotlib import pyplot as plt, animation
import numpy as np

from environments import Environment


class Pendulum(Environment):
    """Pendulum System

    Equations of movement are:

        theta'' = -(g/l)*sin(theta)

    """
    def __init__(self, mass, length, g, q=None, p=None):
        """Constructor for pendulum system

        Args:
            mass (float): Pendulum mass (kg)
            length (float): Pendulum length (m)
            g (float): Gravity of the environment (m/s^2)
            q ([float], optional): Generalized position in 1-D space: Phase (rad). Defaults to None
            p ([float], optional): Generalized momentum in 1-D space: Angular momentum (kg*m^2/s). Defaults to None
        """
        self.mass = mass
        self.length = length
        self.g = g
        super().__init__(q=q, p=p)

    def set(self, q, p):
        """Sets initial conditions for pendulum

        Args:
            q ([float]): Generalized position in 1-D space: Phase (rad)
            p ([float]): Generalized momentum in 1-D space: Angular momentum (kg*m^2/s)

        Raises:
            ValueError: If p and q are not in 1-D space
        """
        if q is None or p is None:
            return
        if len(q) != 1 or len(p) != 1:
            raise ValueError(
                "q and p must be in 1-D space: Angular momentum and Phase.")
        self.q = q
        self.p = p

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        return [(states[1] / (self.mass * self.length * self.length)),
                -self.g * self.mass * self.length * np.sin(states[0])]

    def _draw(self, res=32, color=True):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square)
            color (bool): True if RGB, false if grayscale 

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self._rollout[0, :]
        length = len(q)
        if color:
            vid = np.zeros((length, res, res, 3), dtype='float')
        else:
            vid = np.zeros((length, res, res, 1), dtype='float')
        SIZE = 1.5
        grid = np.arange(0, 1, 1. / res) * 2 * SIZE - SIZE
        [I, J] = np.meshgrid(grid, grid)
        for t in range(length):
            if color:
                vid[t, :, :, 0] += np.exp(-(((I - np.sin(q[t]))**2 +
                                             (J - np.cos(q[t]))**2) /
                                            (self.mass**2))**4)
                vid[t, :, :, 1] += np.exp(-(((I - np.sin(q[t]))**2 +
                                             (J - np.cos(q[t]))**2) /
                                            (self.mass**2))**4)
            else:
                vid[t, :, :, 0] += np.exp(-(((I - np.sin(q[t]))**2 +
                                             (J - np.cos(q[t]))**2) /
                                            (self.mass**2))**4)
            vid[t][vid[t] > 1] = 1

        return vid

    def _sample_init_conditions(self, radius):
        """Samples random initial conditions for the environment

        Args:
            radius (float): Radius of the sampling process
        """
        states = np.random.rand(2) * 2. - 1
        states /= np.sqrt((states**2).sum()) * radius
        self.set([states[0]], [states[1]])


# Sample code for sampling rollouts
if __name__ == "__main__":

    pd = Pendulum(mass=.5, length=1, g=3)
    rolls = pd.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_std=0.,
                                      seed=23)
    fig = plt.figure()
    img = []
    idx = np.random.randint(rolls.shape[0])
    for im in rolls[idx]:
        img.append([plt.imshow(im, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=50,
                                    blit=True,
                                    repeat_delay=1000)
    plt.show()
