import cv2
import numpy as np

from environment import Environment, visualize_rollout


class ChaoticPendulum(Environment):
    """Chaotic Pendulum System: 2 Objects

    Hamiltonian system is:

        H = (1/2*m*L^2)* (p_1^2 + 2*p_2^2 - 2*p_1*p_2* \
             cos(q_1 - q_2)) / (1 + sin^2(q_1 - q_2))
            + mgL*(3 - 2*cos q_1 - cos q_2)

    """

    WORLD_SIZE = 2.5

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
        if len(q) != 2 or len(p) != 2:
            raise ValueError(
                "q and p must be 2 objects in 1-D space: Angular momentum and Phase."
            )
        self.q = q
        self.p = p

    def get_world_size(self):
        """Return world size for correctly render the environment.
        """
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        return 0.05

    def get_default_radius_bounds(self):
        """Returns:
            radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        return (0.5, 1.3)

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        states_resh = states.reshape(2, 2)
        dyn = np.zeros_like(states_resh)

        # dq_1 and dq_2
        quot = self.mass*(self.length**2) * \
            (1 + (np.sin(states_resh[0, 0] - states_resh[0, 1])**2))
        dyn[0, 0] = states_resh[1, 0] - states_resh[1, 1] * \
            np.cos(states_resh[0, 0] - states_resh[0, 1])
        dyn[0, 1] = states_resh[1, 1] - states_resh[1, 0] * \
            np.cos(states_resh[0, 0] - states_resh[0, 1])
        dyn[0, :] /= quot
        # dp_1 and dp_2
        dyn[1, :] -= 2 * self.mass * self.g * self.length * np.sin(
            states_resh[0, :])
        cst = 1 / (2 * self.mass * (self.length**2))
        term1 = states_resh[1, 0]**2 + states_resh[1, 1]**2 + \
            2*states_resh[1, 0]*states_resh[1, 1] * \
            np.cos(states_resh[0, 0] - states_resh[0, 1])
        term2 = (1 + (np.sin(states_resh[0, 0] - states_resh[0, 1])**2))

        dterm1_dq_1 = 2*states_resh[1, 0]*states_resh[1, 1] * \
            np.sin(states_resh[0, 0] - states_resh[0, 1])
        dterm1_dq_2 = -dterm1_dq_1

        dterm2_dq_1 = 2 * np.cos(states_resh[0, 0] - states_resh[0, 1])
        dterm2_dq_2 = -dterm2_dq_1

        dyn[1, 0] -= cst * (dterm1_dq_1 * term2 - term1 *
                            dterm2_dq_1) / (term2 ** 2)
        dyn[1, 1] -= cst * (dterm1_dq_2 * term2 - term1 *
                            dterm2_dq_2) / (term2 ** 2)

        return dyn.reshape(-1)

    def _draw(self, res=32, color=True):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self._rollout.reshape(2, 2, -1)[0, :, :]
        length = q.shape[-1]
        vid = np.zeros((length, res, res, 3), dtype='float')
        ball_colors = self._default_ball_colors
        space_res = 2.*self.get_world_size()/res
        for t in range(length):
            coords_1 = self._world_to_pixels(
                self.length * np.sin(q[0, t]), self.length * np.cos(q[0, t]), res)
            coords_2 = self._world_to_pixels(
                self.length * np.sin(q[0, t]) + self.length * np.sin(q[1, t]),
                self.length * np.cos(q[0, t]) + self.length * np.cos(q[1, t]),
                res)
            vid[t] = cv2.circle(vid[t], coords_1, int(
                self.length/(space_res*3)), ball_colors[0], -1)
            vid[t] = cv2.circle(vid[t], coords_2, int(
                self.length/(space_res*3)), ball_colors[1], -1)
            vid[t] = cv2.blur(cv2.blur(vid[t], (2, 2)), (2, 2))
        vid += self._default_background_color
        vid[vid > 1.] = 1.
        if not color:
            vid = np.expand_dims(np.max(vid, axis=-1), -1)
        return vid

    def _sample_init_conditions(self, radius):
        """Samples random initial conditions for the environment

        Args:
            radius (float): Radius of the sampling process
        """
        states_q = np.random.rand(2) * 2. - 1
        states_q = (states_q / np.sqrt((states_q**2).sum())) * radius
        states_p = np.random.rand(2) * 2. - 1
        states_p = (states_p / np.sqrt((states_p**2).sum())) * radius
        self.set(states_q, states_p)


# Sample code for sampling rollouts
if __name__ == "__main__":

    pd = ChaoticPendulum(mass=1., length=1, g=3)
    rolls = pd.sample_random_rollouts(number_of_frames=300,
                                      delta_time=.125,
                                      number_of_rollouts=1,
                                      img_size=64,
                                      noise_level=0.,
                                      radius_bound=(0.5, 1.3),
                                      seed=None)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
