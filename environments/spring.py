from matplotlib import pyplot as plt, animation
import numpy as np

from environments import Environment


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
            p ([float], optional): Generalized momentum in 1-D space: Linear momentum (kg*m/s). Defaults to None
            q ([float], optional): Generalized position in 1-D space: Position (m). Defaults to None
        """
        self.mass = mass
        self.elastic_cst = elastic_cst
        super().__init__(p, q)

    def set(self, p, q):
        """Sets initial conditions for spring system

        Args:
            p ([float]): Generalized momentum in 1-D space: Linear momentum (kg*m/s)
            q ([float]): Generalized position in 1-D space: Position (m)

        Raises:
            AssertError: If p and q are not in 1-D space
        """
        if p is None or q is None:
            return
        if len(p) != 1 or len(q) != 1:
            raise ValueError(
                "p and q must be in 1-D space: Angular momentum and Phase.")
        self.p = p
        self.q = q

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        return [states[1]/self.mass, -self.elastic_cst*states[0]]

    def _draw(self, res=32, color=True):
        """Returns array of the environment evolution

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
        grid = np.arange(0, 1, 1./res)*2*SIZE - SIZE
        [I, J] = np.meshgrid(grid, grid)
        for t in range(length):
            if color:
                vid[t, :, :, 0] += np.exp(-(((I - 0)**2 +
                                             (J - q[t])**2) / (self.mass**2))**4)
                vid[t, :, :, 1] += np.exp(-(((I - 0)**2 +
                                             (J - q[t])**2) / (self.mass**2))**4)
            else:
                vid[t, :, :, 0] += np.exp(-(((I - 0)**2 +
                                             (J - q[t])**2) / (self.mass**2))**4)
            vid[t][vid[t] > 1] = 1

        return vid

    def sample_init_conditions(self, radius):
        """Samples random initial conditions for the environment

        Args:
            radius (float): Radius of the sampling process
        """
        states = np.random.rand(2)*2.-1
        states = (states/np.sqrt((states**2).sum()))*radius
        self.set([states[0]], [states[1]])


# Sample code for sampling rollouts
if __name__ == "__main__":

    sp = Spring(mass=.5, elastic_cst=2)
    rolls = sp.sample_random_rollouts(number_of_frames=100, delta_time=0.1,
                                      number_of_rollouts=16, img_size=32,
                                      noisy_data=False, noise_std=0.1,
                                      radius_lb=0.1, radius_ub=1.0, seed=23)
    fig = plt.figure()
    img = []
    idx = np.random.randint(rolls.shape[0])
    for im in rolls[idx]:
        img.append([plt.imshow(im, animated=True)])
    ani = animation.ArtistAnimation(fig, img, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()
