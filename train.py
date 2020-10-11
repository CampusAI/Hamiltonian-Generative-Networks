"""Script to train the Hamiltonian Generative Network
"""
import torch

from hamiltonian_generative_network import HGN
import environments
import networks
import utils

# Data Parameters
sequence_length = 5
delta_t = 0.1

# Network Parameters
encoder_out_channels = 48
transformer_out_channels = 32

# Optimization Parameters
training_steps = 1
encoder_lr = 1e-2
transformer_lr = 1e-2
hnn_lr = 1e-2
momentum_lr = 1e-2
momentum = 0.9

if __name__ == "__main__":
    # Pick environment
    env = environments.test_env.TestEnv()

    # Instantiate networks
    encoder = networks.inference_net.EncoderNet(
        seq_len=sequence_length, out_channels=encoder_out_channels)
    transformer = networks.inference_net.EncoderNet(
        in_channels=encoder_out_channels,
        out_channels=transformer_out_channels)
    hnn = networks.hamiltonian_net.HamiltonianNet(
        in_channels=transformer_out_channels)
    decoder = None

    # Define HGN integrator
    integrator = utils.integrator.Integrator(delta_t=delta_t, method="Euler")

    # Define optimization modules
    optim_params = [
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': transformer.parameters(), 'lr': transformer_lr},
        {'params': hnn.parameters(), 'lr': hnn_lr},
        {'params': momentum.parameters(), 'lr': momentum_lr},
    ]
    optimizer = torch.optim.SGD(optim_params, momentum=momentum)
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
              channels=1)

    for i in range(training_steps):  # For each training step
        # Sample rollouts from the environment  TODO(oleguer): Move this into a dataloader
        rollouts = env.sample_random_rollout(seed=i,
                                             number_of_frames=5,
                                             delta_time=delta_t,
                                             number_of_rollouts=1)
        hgn.fit(rollouts)