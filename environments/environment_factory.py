"""
"""
from environments import Environment
# TODO(Oleguer): Fix this, its not very nice to have to import all classes
from pendulum import Pendulum
# from spring import Spring


class EnvFactory():
    """Return a new Environment"""

    # Map the name of the name of the Environment concrete class by retrieving all its subclasses
    _name_to_env = {cl.__name__: cl for cl in Environment.__subclasses__()}

    @staticmethod
    def get_environment(name, **kwargs):
        """Return an environment object based on the environment identifier.

        Args:
            name (string); name of the class of the concrete Environment.
            **kwargs: args supplied to the constructor of the object of class name. 
        
        Raises:
            (NameError): if the given environment type is not supported.
        
        Returns:
            (Environment): concrete instantiation of the Environment.
        """
        try:
            return EnvFactory._name_to_env[name](**kwargs)
        except KeyError:
            msg = "%s is not a supported type by Environment." % (name)
            msg += "Available types are: " + "".join("%s " % eef for eef in EnvFactory._name_to_env.keys())
            raise NameError(msg)

if __name__ == "__main__":
    # EnvFactory test
    env = EnvFactory.get_environment("Pendulum", mass=0.5, length=1, g=10)
    print(type(env))

    from matplotlib import pyplot as plt, animation
    import numpy as np
    rolls = env.sample_random_rollouts(number_of_frames=100,
                                        delta_time=0.1,
                                        number_of_rollouts=16,
                                        img_size=32,
                                        noise_std=0.,
                                        seed=23)
    fig = plt.figure()
    img = []
    idx = np.random.randint(rolls.shape[0])
    for im in rolls[idx]:
        img.append([plt.imshow(im, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=50,
                                    blit=True,
                                    repeat_delay=1000)
    plt.show()