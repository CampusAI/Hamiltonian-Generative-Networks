import numpy as np

from environment import Environment, visualize_rollout


class Spring(Environment):
    """Damped spring System

    Equations of movement are:

        x'' = -2*c*sqrt(k/m)*x' -(k/m)*x  

    """

    WORLD_SIZE = 2.

    def __init__(self, mass, elastic_cst, damping_ratio=0., q=None, p=None):
        """Constructor for spring system

        Args:
            mass (float): Spring mass (kg)
            elastic_cst (float): Spring elastic constant (kg/s^2)
            damping_ratio (float): Damping ratio of the oscillator
                if damping_ratio > 1: Oscillator is overdamped
                if damping_ratio = 1: Oscillator is critically damped
                if damping_ratio < 1: Oscillator is underdamped
            q ([float], optional): Generalized position in 1-D space: Position (m). Defaults to None
            p ([float], optional): Generalized momentum in 1-D space: Linear momentum (kg*m/s). Defaults to None
        """
        self.mass = mass
        self.elastic_cst = elastic_cst
        self.damping_ratio = damping_ratio
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

    def get_world_size(self):
        """Return world size for correctly render the environment.
        """
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        return 0.1

    def get_default_radius_bounds(self):
        """Returns:
            radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        return (0.1, 1.0)

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        # angular freq of the undamped oscillator
        w0 = np.sqrt(self.elastic_cst/self.mass)
        # dynamics of the damped oscillator
        return [states[1] / self.mass, -2*self.damping_ratio*w0*states[1] - self.elastic_cst*states[0]]

    def _draw(self, res=32, color=True):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.

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
        grid = np.arange(0, 1, 1. / res) * 2 * \
            self.WORLD_SIZE - self.WORLD_SIZE
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
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.
        """
        radius_lb, radius_ub = radius_bound
        radius = np.random.rand()*(radius_ub - radius_lb) + radius_lb
        states = np.random.rand(2) * 2. - 1
        states = (states / np.sqrt((states**2).sum())) * radius
        self.set([states[0]], [states[1]])


# Sample code for sampling rollouts
if __name__ == "__main__":

    sp = Spring(mass=.5, elastic_cst=2, damping_ratio=.1)
    rolls = sp.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(.5, 1.4),
                                      seed=None)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
