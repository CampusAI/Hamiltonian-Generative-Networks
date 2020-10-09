import torch


class Integrator:
    METHODS = ["Euler", "RK4", "leapfrog"]

    def __init__(self, delta_t, method="Euler"):
        if method not in self.METHODS:
            msg = "%s is not a supported method. " % (method)
            msg += "Available methods are: " + "".join("%s " % m for m in self.METHODS)
            raise KeyError(msg)

        self.delta_t = delta_t
        self.method = method

    def _get_grads(self, q, p, hnn):
        # Compute energy of the system
        energy = hnn(q=q, p=p)

        # Keep non-leaf gradients
        q.retain_grad()
        p.retain_grad()
        energy.backward(retain_graph=True)  # Compute dH/dq, dH/dp

        # Hamilton formulas
        dq_dt = p.grad  # dq_dt = dH/dp
        dp_dt = -q.grad  # dp_dt = -dH/dq

        return dq_dt, dp_dt

    def _euler_step(self, q, p, hnn):
        dq_dt, dp_dt = self._get_grads(q, p, hnn)

        # Euler integration
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return q_next, p_next

    def _rk_step(self, q, p, hnn):
        # k1
        k1_q, k1_p = self._get_grads(q, p, hnn)

        # k2
        q_2 = q + self.delta_t * k1_q / 2  # x = x_t + dt * k1 / 2
        p_2 = p + self.delta_t * k1_p / 2  # x = x_t + dt * k1 / 2
        k2_q, k2_p = self._get_grads(q_2, p_2, hnn)

        # k3
        q_3 = q + self.delta_t * k2_q / 2  # x = x_t + dt * k2 / 2
        p_3 = p + self.delta_t * k2_p / 2  # x = x_t + dt * k2 / 2
        k3_q, k3_p = self._get_grads(q_3, p_3, hnn)

        # k4
        q_3 = q + self.delta_t * k3_q / 2  # x = x_t + dt * k3
        p_3 = p + self.delta_t * k3_p / 2  # x = x_t + dt * k3
        k4_q, k4_p = self._get_grads(q_3, p_3, hnn)

        # Runge-Kutta 4 integration
        q_next = q + self.delta_t * ((k1_q/6) + (k2_q/3) + (k3_q/3) + (k4_q/6))
        p_next = p + self.delta_t * ((k1_p/6) + (k2_p/3) + (k3_p/3) + (k4_p/6))
        return q_next, p_next


    def step(self, q, p, hnn):
        if self.method == "Euler":
            return self._euler_step(q, p, hnn)
        raise NotImplementedError
