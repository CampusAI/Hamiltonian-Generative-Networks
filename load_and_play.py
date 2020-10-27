import yaml
import torch

from utilities import loader
import hamiltonian_generative_network
from utilities import integrator
from environments import pendulum
from utilities import conversions


def _read_config(path):
    with open(path) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


if __name__ == '__main__':
    SEQ_LEN = 60
    N_STEPS = 30
    DELTA_TIME = 0.125
    MODEL_TO_LOAD = 'saved_models/split'
    DEFAULT_PARAMS = _read_config('experiment_params/train_config_default.yaml')

    env = pendulum.Pendulum(mass=0.5, length=1., g=3.)

    leapfrog = integrator.Integrator(delta_t=0.125, method='Leapfrog')
    hgn = hamiltonian_generative_network.HGN(integrator=leapfrog, seq_len=SEQ_LEN, device='cpu',
                                             dtype=torch.float)
    hgn.load(MODEL_TO_LOAD)
    hgn.potential = loader.instantiate_potential(DEFAULT_PARAMS, 'cpu', torch.float)

    rollout = env.sample_random_rollouts(
        number_of_frames=SEQ_LEN, delta_time=DELTA_TIME, number_of_rollouts=1, img_size=32,
        radius_bound='auto', noise_level=0.0,
    )
    rollout = torch.from_numpy(rollout)
    rollout = conversions.to_channels_first(rollout).type(torch.float)

    res = hgn.forward(rollout[:, :N_STEPS])
    res.visualize(interval=400, show_step=True)
