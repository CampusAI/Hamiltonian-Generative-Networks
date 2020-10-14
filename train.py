"""Script to train the Hamiltonian Generative Network
"""
import torch
import yaml

from environments.environment_factory import EnvFactory
from hamiltonian_generative_network import HGN
from networks.inference_net import EncoderNet, TransformerNet
from networks.hamiltonian_net import HamiltonianNet
from networks.decoder_net import DecoderNet
import utilities

params_file = "experiment_params/default.yaml"

if __name__ == "__main__":
    # Read parameters
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Pick environment
    env = EnvFactory.get_environment(**params["environment"])

    # Instantiate networks
    encoder = EncoderNet(seq_len=params["rollout"]["seq_length"],
                         in_channels=params["rollout"]["n_channels"],
                         **params["networks"]["encoder"])
    transformer = TransformerNet(
        in_channels=params["networks"]["encoder"]["out_channels"],
        **params["networks"]["transformer"])
    hnn = HamiltonianNet(**params["networks"]["hamiltonian"])
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"] / 2,
        out_channels=params["rollout"]["n_channels"],
        **params["networks"]["decoder"])

    # Define HGN integrator
    integrator = utilities.integrator.Integrator(
        delta_t=params["rollout"]["delta_time"],
        method=params["integrator"]["method"])

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
    optimizer = torch.optim.Adam(optim_params)
    loss = torch.nn.MSELoss()

    # Instantiate Hamiltonian Generative Network
    hgn = HGN(encoder=encoder,
              transformer=transformer,
              hnn=hnn,
              decoder=decoder,
              integrator=integrator,
              loss=loss,
              optimizer=optimizer,
              seq_len=params["rollout"]["seq_length"],
              channels=params["rollout"]["n_channels"])

    for i in range(params["optimization"]["training_steps"]):  # For each training step
        # Sample rollouts from the environment NOTE: This will be changed into the dataloader once its implemented
        rollouts = env.sample_random_rollouts(
            seed=i,
            number_of_frames=params["rollout"]["seq_length"],
            delta_time=params["rollout"]["delta_time"],
            number_of_rollouts=params["optimization"]["batch_size"],
            color=(params["rollout"]["n_channels"]==3))

        error = hgn.fit(rollouts)
        break