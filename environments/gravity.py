import numpy as np

from environment import Environment, visualize_rollout


class NObjectGravity(Environment):

    """N Object Gravity Atraction System

    Equations of movement are:

        m_i*q'' = G * sum_{i!=j} ( m_i*m_j*(q_j - q_i)/ abs(q_j  - q_i)^3 )

    """

    WORLD_SIZE = 6.

    def __init__(self, mass, g, orbit_noise=.01, q=None, p=None):
        """Contructor for spring system

        Args:
            mass (list): List of floats corresponding to object masses (kg).
            g (float): Constant for the intensity of gravitational field (m^3/kg*s^2)
            orbit_noise (float, optional): Noise for object orbits when sampling initial conditions
            q (ndarray, optional): Object generalized positions in 2-D space: Positions (m). Defaults to None
            p (ndarray, optional): Object generalized momentums in 2-D space : Linear momentums (kg*m/s). Defaults to None
        Raises:
            NotImplementedError: If more than 7 objects are considered
        """
        self.mass = mass
        self.colors = ['y', 'r', 'g', 'b', 'c', 'p', 'w']
        self.n_objects = len(mass)
        self.g = g
        self.orbit_noise = orbit_noise
        if self.n_objects > 7:
            raise NotImplementedError(
                'Gravity interaction for ' + str(self.n_objects) + ' bodies is not implemented.')
        super().__init__(q=q, p=p)

    def set(self, q, p):
        """Sets initial conditions for pendulum

        Args:
            q (ndarray): Object generalized positions in 2-D space: Positions (m)
            p (ndarray): Object generalized momentums in 2-D space : Linear momentums (kg*m/s)

        Raises:
            ValueError: If q and p are not in 2-D space or do not refer to all the objects in space
        """
        if q is None or p is None:
            return
        if q.shape[0] != self.n_objects or p.shape[0] != self.n_objects:
            raise ValueError(
                "q and p do not refer to the same number of objects in the system.")
        if q.shape[-1] != 2 or p.shape[-1] != 2:
            raise ValueError(
                "q and p must be in 2-D space: Position and Linear momentum.")
        self.q = q.copy()
        self.p = p.copy()

    def get_world_size(self):
        """Return world size for correctly render the environment.
        """
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        if self.n_objects == 2:
            return 0.05
        elif self.n_objects == 3:
            return 0.005
        else:
            return 0.

    def get_default_radius_bounds(self):
        """Returns:
            radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        if self.n_objects == 2:
            return (0.5, 1.5)
        elif self.n_objects == 3:
            return (0.9, 1.2)
        else:
            return (1., 1.)  # TODO: Should we raise an error?

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states (numpy.ndarray): 1-D array that contains the information of the phase
                state, in the format of np.array([q,p]).reshape(-1).

        Returns:
            equations (numpy.ndarray): Numpy array with derivatives of q and p w.r.t. time
        """
        # Convert states to n_object arrays of q and p
        states_resh = states.reshape(2, self.n_objects, 2)
        dyn = np.zeros_like(states_resh)
        states_q = states_resh[0, :, :]
        states_p = states_resh[1, :, :]
        dyn[0, :, :] = states_p/(np.array(self.mass)[:, np.newaxis])

        # Distance calculation
        object_distance = np.zeros((self.n_objects, self.n_objects))
        for i in range(self.n_objects):
            for j in range(i, self.n_objects):
                object_distance[i, j] = np.linalg.norm(
                    states_q[i] - states_q[j])
                object_distance[j, i] = object_distance[i, j]
        object_distance = np.power(object_distance, 3)/self.g

        for d in range(2):
            for i in range(self.n_objects):
                mom_term = 0
                for j in range(self.n_objects):
                    if i != j:
                        mom_term += self.mass[j]*(states_q[j, d] -
                                                  states_q[i, d])/object_distance[i, j]
                dyn[1, i, d] += mom_term*self.mass[i]
        return dyn.reshape(-1)

    def _draw(self, res=32, color=True):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.

        Returns:
            vid (np.ndarray): Numpy array of shape (seq_len, height, width, channels)
                containing the rendered rollout as a sequence of images.
        """
        q = self._rollout.reshape(2, self.n_objects, 2, -1)[0]
        length = q.shape[-1]
        if color:
            vid = np.zeros((length, res, res, 3), dtype='float')
            vid += 80./255.
        else:
            vid = np.zeros((length, res, res, 1), dtype='float')
        grid = np.arange(0, 1, 1./res) * 2 * self.WORLD_SIZE - self.WORLD_SIZE
        [I, J] = np.meshgrid(grid, grid)
        for t in range(length):
            if color:
                for n in range(self.n_objects):
                    brush = self.colors[n]
                    color_term = np.exp(-(((I - q[n, 0, t])**2 +
                                           (J - q[n, 1, t])**2) /
                                          (self.mass[n]**2))**4)
                    if brush == 'y':
                        vid[t, :, :, 0] += color_term
                        vid[t, :, :, 1] += color_term
                        vid[t, :, :, 2] -= color_term
                    elif brush == 'r':
                        vid[t, :, :, 0] += color_term
                        vid[t, :, :, 1] -= color_term
                        vid[t, :, :, 2] -= color_term
                    elif brush == 'g':
                        vid[t, :, :, 0] -= color_term
                        vid[t, :, :, 1] += color_term
                        vid[t, :, :, 2] -= color_term
                    elif brush == 'b':
                        vid[t, :, :, 0] -= color_term
                        vid[t, :, :, 1] -= color_term
                        vid[t, :, :, 2] += color_term
                    elif brush == 'c':
                        vid[t, :, :, 0] -= color_term
                        vid[t, :, :, 1] += color_term
                        vid[t, :, :, 2] += color_term
                    elif brush == 'p':
                        vid[t, :, :, 0] += color_term
                        vid[t, :, :, 1] -= color_term
                        vid[t, :, :, 2] += color_term
                    elif brush == 'w':
                        vid[t, :, :, 0] += color_term
                        vid[t, :, :, 1] += color_term
                        vid[t, :, :, 2] += color_term
                    vid[t][vid[t] < 0] = 0
            else:
                for i in range(self.n_objects):
                    vid[t, :, :, 0] += np.exp(-(((I - q[i, 0, t])**2 +
                                                 (J - q[i, 1, t])**2) /
                                                (self.mass[i]**2))**4)
            vid[t][vid[t] > 1] = 1
            vid[t][vid[t] < 0] = 0

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
        states = np.zeros((2, self.n_objects, 2))
        # first position
        pos = np.random.rand(2)*2. - 1
        pos = (pos/np.sqrt((pos**2).sum()))*radius

        # velocity that yields a circular orbit
        if self.n_objects == 2:
            factor = 2
        else:
            factor = np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))
        vel = np.flipud(pos)/(factor*radius**1.5)
        vel[0] *= -1
        states[0, 0, :] = pos
        states[1, 0, :] = vel

        rot_angle = 2*np.pi/self.n_objects
        for i in range(1, self.n_objects):
            states[0, i, :] = self.__rotate2d(
                states[0, i - 1, :], theta=rot_angle)
            states[1, i, :] = self.__rotate2d(
                states[1, i - 1, :], theta=rot_angle)
        for i in range(self.n_objects):
            states[1, i, :] *= 1 + \
                self.orbit_noise*(2*np.random.rand(2) - 1)
        self.set(states[0], states[1])

    def __rotate2d(self, p, theta):
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array([[c, -s], [s, c]])
        return np.dot(Rot, p.reshape(2, 1)).squeeze()


# Sample code for sampling rollouts
if __name__ == "__main__":

    og = NObjectGravity(mass=[1., 1., 1.],
                        g=1., orbit_noise=0.1)
    rolls = og.sample_random_rollouts(number_of_frames=1000,
                                      delta_time=0.1,
                                      number_of_rollouts=1,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(2., 3.2),
                                      seed=33)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
