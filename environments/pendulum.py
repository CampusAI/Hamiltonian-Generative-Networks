import cv2
import numpy as np

from environment import Environment, visualize_rollout


class Pendulum(Environment):
    """Pendulum System

    Equations of movement are:

        theta'' = -(g/l)*sin(theta)

    """

    WORLD_SIZE = 2.

    def __init__(self, mass, length, g, q=None, p=None):
        """Constructor for pendulum system

        Args:
            mass (float): Pendulum mass (kg)
            length (float): Pendulum length (m)
            g (float): Gravity of the environment (m/s^2)
            q ([float], optional): Generalized position in 1-D space: Phase (rad). Defaults to None
            p ([float], optional): Generalized momentum in 1-D space: Angular momentum (kg*m^2/s).
                Defaults to None
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
        return (1.3, 2.3)

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
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self._rollout[0, :]
        length = len(q)
        vid = np.zeros((length, res, res, 3), dtype='float')
        ball_color = self._default_ball_colors[0]
        space_res = 2.*self.get_world_size()/res
        for t in range(length):
            vid[t] = cv2.circle(vid[t], self._world_to_pixels(self.length*np.sin(q[t]),
                                                              self.length*np.cos(q[t]), res),
                                int(self.mass/space_res), ball_color, -1)
            vid[t] = cv2.blur(cv2.blur(vid[t], (2, 2)), (2, 2))
        vid += self._default_background_color
        vid[vid > 1.] = 1.
        if not color:
            vid = np.expand_dims(np.max(vid, axis=-1), -1)
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

    pd = Pendulum(mass=.5, length=1, g=3)
    rolls = pd.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(1.3, 2.3),
                                      color=True,
                                      seed=23)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
