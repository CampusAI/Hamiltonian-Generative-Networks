"""
"""
from environments import Environment
# TODO(Oleguer): Fix this, its not very nice to have to import all classes
from pendulum import Pendulum
from spring import Spring


class EnvFactory():
    """Return a new Environment"""

    # Map the name of the name of the Environment concrete class by retrieving all its subclasses
    _name_to_env = {cl.__name__: cl for cl in Environment.__subclasses__()}

    @staticmethod
    def get_environment(class_name, **kwargs):
        """Return an environment object based on the environment identifier.

        Args:
            class_name (string); name of the class of the concrete Environment.
            **kwargs: args supplied to the constructor of the object of class class_name. 
        
        Raises:
            (NameError): if the given environment type is not supported.
        
        Returns:
            (Environment): concrete instantiation of the Environment.
        """
        try:
            return EnvFactory._name_to_env[class_name](**kwargs)
        except KeyError:
            msg = "%s is not a supported type by Environment." % (class_name)
            msg += "Available types are: " + "".join("%s " % eef for eef in EnvFactory._name_to_env.keys())
            raise NameError(msg)

if __name__ == "__main__":
    # EnvFactory test
    env = EnvFactory.get_environment("Pendulum", mass=1, length=1)
    print(type(env))