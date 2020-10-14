from .environments import Environment

import numpy as np


class NObjectGravity(Environment):

    """N Object Gravity Atraction System

    Equations of movement are:

        m_i*q'' = G * sum_{i!=j} ( m_i*m_j*(q_j - q_i)/ abs(q_j  - q_i)^3 )

    """

    def __init__(self, mass, gravity_cst, q=None, p=None):
        """Contructor for spring system

        Args:
            mass ([float]): Object masses (kg).
            gravity_cst (float): Constant for the intensity of gravitational field (m^3/kg*s^2)
            q (ndarray, optional): Object generalized positions in 2-D space: Positions (m). Defaults to None
            p (ndarray, optional): Object generalized momentums in 2-D space : Linear momentums (kg*m/s). Defaults to None
        """
        self.mass = mass
        self.n_objects = len(mass)
        self.gravity_cst = gravity_cst
        super().__init__(q=q, p=p)

    def set(self, q, p):
        """Sets initial conditions for pendulum

        Args:
            q (ndarray): Object generalized positions in 2-D space: Positions (m)
            p (ndarray): Object generalized momentums in 2-D space : Linear momentums (kg*m/s)

        Raises:
            ValueError: If q and p are not in 2-D space or do not refer to all the objects in space
        """
        if q.shape[0] != self.n_objects or p.shape[0] != self.n_objects:
            raise ValueError(
                "q and p do not refer to the same number of objects in the system.")
        if q.shape[-1] != 2 or p.shape[-1] != 2:
            raise ValueError(
                "q and p must be in 2-D space: Position and Linear momentum.")
        self.q = q.copy()
        self.p = p.copy()

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        # Convert states to n_object arrays of q and p
        states_resh = states.reshape(2, self.n_objects, 2)
        dyn = np.zeros_like(states_resh)
        states_q = states_resh[0, :, :]
        states_p = states_resh[1, :, :]
        dyn[0, :, :] = states_p/(np.array(mass)[:, np.newaxis])
        for d in range(2):
            for i in range(self.n_objects):
                for j in range(self.n_objects):
                    if i != j:
                        mom_term += self.mass[j]*(self.q[j, d] -
                                                  self.q[i, d])/object_distance[i, j]
                dyn[1, :, :] += mom_term*self.mass[i]
        return dyn.reshape(-1)

    def _draw(self, res=32, color=True, world_size=1.5):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.
            world_size (float) Spatial extent of the window where the rendering is taking place (in meters).

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self._rollout.reshape(2, self.n_objects, 2, -1)[0, :, :, :]
        length = len(q)
        if color:
            vid = np.zeros((length, res, res, 3), dtype='float')
        else:
            vid = np.zeros((length, res, res, 1), dtype='float')
        grid = np.arange(0, 1, 1./res)*2*world_size - world_size
        [I, J] = np.meshgrid(grid, grid)
        for t in range(length):
            if color:
                vid[t, :, :, 0] += np.exp(-(((I - q[0, 0, t])**2 +
                                             (J - q[0, 1, t])**2) /
                                            (self.mass**2))**4)
                vid[t, :, :, 1] += np.exp(-(((I - q[1, 0, t])**2 +
                                             (J - q[1, 1, t])**2) /
                                            (self.mass**2))**4)
                if self.n_objects > 2:
                    vid[t, :, :, 2] += np.exp(-(((I - q[2, 0, t])**2 +
                                                 (J - q[2, 1, t])**2) /
                                                (self.mass**2))**4)
            else:
                for i in range(self.n_objects):
                    vid[t, :, :, 0] += np.exp(-(((I - q[i, 0, t])**2 +
                                                 (J - q[i, 1, t])**2) /
                                                (self.mass**2))**4)
            vid[t][vid[t] > 1] = 1

        return vid

    def _sample_init_conditions(self, radius):
        """Samples random initial conditions for the environment
        Args:
            radius (float): Radius of the sampling process
        """
        states = np.random.rand(2) * 2. - 1
        states = (states / np.sqrt((states**2).sum())) * radius
        self.set([states[0]], [states[1]])


# Sample code for sampling rollouts
if __name__ == "__main__":

    og = NObjectGravity(mass=[1., 1.], gravity_cst=1)
    rolls = og.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_std=0.,
                                      radius_bound=(1.3, 2.3),
                                      world_size=1.5,
                                      seed=23)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
