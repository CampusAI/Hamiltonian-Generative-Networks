from .environments import Environment

import numpy as np


class NObjectGravity(Environment):

    """N Object Gravity Atraction System

    Equations of movement are:

        m_i*q'' = G * sum_{i!=j} ( m_i*m_j*(q_j - q_i)/ abs(q_j  - q_i)^3 )

    """

    def __init__(self, mass, gravity_cst, p=None, q=None):
        """Contructor for spring system

        Args:
            mass ([float]): Object masses (kg).
            gravity_cst (float): Constant for the intensity of gravitational field (m^3/kg*s^2)
            p (ndarray, optional): Object generalized momentums in 2-D space : Linear momentums (kg*m/s). Defaults to None
            q (ndarray, optional): Object generalized positions in 2-D space: Positions (m). Defaults to None
        """
        self.mass = mass
        self.n_objects = len(mass)
        self.gravity_cst = gravity_cst
        super().__init__(p, q)

    def set(self, p, q):
        """Sets initial conditions for pendulum

        Args:
            p (ndarray): Object generalized momentums in 2-D space : Linear momentums (kg*m/s)
            q (ndarray): Object generalized positions in 2-D space: Positions (m)

        Raises:
            ValueError: If p and q are not in 2-D space or do not refer to all the objects in space
        """
        if p.shape[0] != self.n_objects or p.shape[0] != self.n_objects:
            raise ValueError(
                "p and q do not refer to the same number of objects in the system.")
        if p.shape[-1] != 2 or q.shape[-1] != 2:
            raise ValueError(
                "p and q must be in 2-D space: Linear momentum and Position.")
        self.p = p.copy()
        self.q = q.copy()

    def step(self, dt=0.01):
        """Performs a step in the spring system

        Args:
            dt (float, optional): Time step run for the integration. Defaults to 0.01.

        Raises:
            TypeError: If p or q are None
        """

        assert type(self.q) != None
        assert type(self.p) != None

        # Distance calculation
        object_distance = np.zeros((self.n_objects, self.n_objects))
        for i in range(self.n_objects):
            for j in range(i, self.n_objects):
                object_distance[i, j] = object_distance[j,
                                                        i] = np.linalg.norm(self.q[i] - self.q[j])
        object_distance = np.power(object_distance, 3)/self.gravity_cst

        for d in range(2):
            for i in range(self.n_objects):
                # update q
                self.q[i, d] += dt*self.p[i, d]/self.mass[i]
                # update momenta
                mom_term = 0
                for j in range(self.n_objects):
                    if i != j:
                        mom_term += self.mass[j]*(self.q[j, d] -
                                                  self.q[i, d])/object_distance[i, j]
                self.p[i, d] += dt*mom_term*self.mass[i]

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

    def draw(self):
        """Caption from the actual spring state

        Returns:
            Img (np.ndarray): Caption of current state
        """
        img = Image.new('L', (32, 32))
        draw = ImageDraw.Draw(img)

        for i in range(self.n_objects):
            r = self.mass[i]
            x = self.q[i, 0] + 32/2
            y = self.q[i, 1] + 32/2

            draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

        return np.array(img)
