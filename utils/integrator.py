class Integrator:
    METHODS = ["Euler", "RK4", "leapfrog"]

    def __init__(self, delta_t, method="Euler"):
        if method in self.METHODS:
            msg = "%s is not a supported method. " % (method)
            msg += "Available methods are: " + "".join("%s " % m for m in self.METHODS)
            raise KeyError(msg)

        self.delta_t = delta_t
        self.method = method

    def _euler_step(self, q, dq_dt, p, dp_dt):
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return q_next, p_next

    def step(self, q, dq_dt, p, dp_dt):
        if self.method == "Euler":
            return self._euler_step(q, dq_dt, p, dp_dt)
        raise NotImplementedError
