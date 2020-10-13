"""Script to train the Hamiltonian Generative Network
"""
import torch
import yaml

from hamiltonian_generative_network import HGN
from environments.environment_factory import EnvFactory
import networks.debug_networks as debug_networks
import utilities

params_file = "experiments_params/default.yaml"

if __name__ == "__main__":
    # Read parameters
    with open(params_file, 'r') as f:
        params = yaml.load(f)

    # Pick environment
    EnvFactory.get_environment(params["environment"])

    # Instantiate networks
    encoder = debug_networks.EncoderNet(seq_len=params["rollout"]["seq_length"],
                                        params["networks"]["encoder"])
    transformer = debug_networks.TransformerNet(params["networks"]["transformer"])
    hnn = debug_networks.HamiltonianNet(params["networks"]["hamiltonian"])
    decoder = debug_networks.DecoderNet(params["networks"]["decoder"])

    # Define HGN integrator
    integrator = utilities.integrator.Integrator(
        delta_t=params["rollout"]["delta_time"], method=params["integrator"]["method"])

    # Define optimization modules
    optim_params = [
        {
            'params': encoder.parameters(),
            'lr': params["optimization"]["encoder_lr"]
        },
        {
            'params': transformer.parameters(),
            'lr': params["optimization"]["transformer_lr"]
        },
        {
            'params': hnn.parameters(),
            'lr': params["optimization"]["hnn_lr"]
        },
        {
            'params': decoder.parameters(),
            'lr': params["optimization"]["decoder_lr"]
        },
    ]
    optimizer = torch.optim.Adam(optim_params, momentum=momentum)
    loss = torch.nn.MSELoss()

    # Instantiate Hamiltonian Generative Network
    hgn = HGN(encoder=encoder,
              transformer=transformer,
              hnn=hnn,
              decoder=decoder,
              integrator=integrator,
              loss=loss,
              optimizer=optimizer,
              seq_len=sequence_length,
              channels=params["rollouts"]["n_channels"])

    for i in range(training_steps):  # For each training step
        # Sample rollouts from the environment  TODO(oleguer): Move this into a dataloader
        rollouts = env.sample_random_rollout(seed=i,
                                             number_of_frames=params["rollout"]["seq_length"],
                                             delta_time=params["rollout"]["delta_time"],
                                             number_of_rollouts=params["optimization"]["batch_size"])
        error = hgn.fit(rollouts)
        break