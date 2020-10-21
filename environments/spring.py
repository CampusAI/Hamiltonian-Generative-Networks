import numpy as np

from environment import Environment, visualize_rollout


class Spring(Environment):
    """Spring System

    Equations of movement are:

        x'' = -(k/m)*x

    """

    def __init__(self, mass, elastic_cst, q=None, p=None):
        """Constructor for spring system

        Args:
            mass (float): Spring mass (kg)
            elastic_cst (float): Spring elastic constant (kg/s^2)
            q ([float], optional): Generalized position in 1-D space: Position (m). Defaults to None
            p ([float], optional): Generalized momentum in 1-D space: Linear momentum (kg*m/s). Defaults to None
        """
        self.mass = mass
        self.elastic_cst = elastic_cst
        super().__init__(q=q, p=p)

    def set(self, q, p):
        """Sets initial conditions for spring system

        Args:
            q ([float]): Generalized position in 1-D space: Position (m)
            p ([float]): Generalized momentum in 1-D space: Linear momentum (kg*m/s)

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
        return [states[1] / self.mass, -self.elastic_cst * states[0]]

    def _draw(self, res=32, color=True, world_size=2.):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.
            world_size (float) Spatial extent of the window where the rendering is taking place (in meters).

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self._rollout[0, :]
        length = len(q)
        if color:
            vid = np.zeros((length, res, res, 3), dtype='float')
            vid += 80./255.
        else:
            vid = np.zeros((length, res, res, 1), dtype='float')
        grid = np.arange(0, 1, 1. / res) * 2 * world_size - world_size
        [I, J] = np.meshgrid(grid, grid)
        for t in range(length):
            if color:
                vid[t, :, :, 0] += np.exp(-(((I - 0)**2 + (J - q[t])**2) /
                                            (self.mass**2))**4)
                vid[t, :, :, 1] += np.exp(-(((I - 0)**2 + (J - q[t])**2) /
                                            (self.mass**2))**4)
            else:
                vid[t, :, :, 0] += np.exp(-(((I - 0)**2 + (J - q[t])**2) /
                                            (self.mass**2))**4)
            vid[t][vid[t] > 1] = 1

        return vid

    def _sample_init_conditions(self, radius_bound):
        """Samples random initial conditions for the environment

        Args:
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
        """
        radius_lb, radius_ub = radius_bound
        radius = np.random.rand()*(radius_ub - radius_lb) + radius_lb
        states = np.random.rand(2) * 2. - 1
        states = (states / np.sqrt((states**2).sum())) * radius
        self.set([states[0]], [states[1]])


# Sample code for sampling rollouts
if __name__ == "__main__":

    sp = Spring(mass=.5, elastic_cst=2)
    rolls = sp.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_std=0.,
                                      radius_bound=(.1, 1.),
                                      seed=23)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
