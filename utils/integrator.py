class Integrator:
    METHODS = ["Euler", "RK4", "leapfrog"]

    def __init__(self, delta_t, method="Euler"):
        if method in self.METHODS:
            msg = "%s is not a supported method. " % (method)
            msg += "Available methods are: " + "".join("%s " % m for m in self.METHODS)
            raise KeyError(msg)

        self.delta_t = delta_t
        self.method = method

    def step(self, q, p, hnn, delta_t):
        if self.method == "Euler":
            hnn
