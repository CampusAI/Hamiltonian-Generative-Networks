"""Script to train the Hamiltonian Generative Network
"""
import torch

from hamiltonian_generative_network import HGN
import environments.test_env as test_env
import networks.debug_networks as debug_networks
import utils

# Data Parameters
sequence_length = 4
delta_t = 0.1

# Network Parameters
encoder_out_channels = 48
transformer_out_channels = 32

# Optimization Parameters
training_steps = 1
encoder_lr = 1e-2
transformer_lr = 1e-2
hnn_lr = 1e-2
decoder_lr = 1e-2
momentum = 0.9

if __name__ == "__main__":
    # Pick environment
    env = test_env.TestEnv()

    # Instantiate networks
    encoder = debug_networks.EncoderNet(seq_len=sequence_length)
    transformer = debug_networks.TransformerNet()
    hnn = debug_networks.HamiltonianNet()
    decoder = debug_networks.DecoderNet()

    # Define HGN integrator
    integrator = utils.integrator.Integrator(delta_t=delta_t, method="Euler")

    # Define optimization modules
    optim_params = [
        {
            'params': encoder.parameters(),
            'lr': encoder_lr
        },
        {
            'params': transformer.parameters(),
            'lr': transformer_lr
        },
        {
            'params': hnn.parameters(),
            'lr': hnn_lr
        },
        {
            'params': decoder.parameters(),
            'lr': decoder_lr
        },
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
        print(rollouts)
        print(rollouts.shape)
        hgn.fit(rollouts)
        break