"""Script to train the Hamiltonian Generative Network
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hamiltonian_generative_network import HGN
import environments.test_env as test_env
import networks.debug_networks as debug_networks
import utilities.integrator as integrator

epsilon = 1e-6

if __name__ == "__main__":
    rollouts = torch.tensor([[[43.23], [22.12], [3.], [4.]]], requires_grad=True).double()

    # Instantiate networks
    encoder = debug_networks.EncoderNet(seq_len=rollouts.shape[1])
    transformer = debug_networks.TransformerNet()
    hnn = debug_networks.HamiltonianNet()
    decoder = debug_networks.DecoderNet()

    # Define HGN integrator
    hgn_integrator = integrator.Integrator(delta_t=0.1, method="Euler")

    # Define optimization module
    optim_params = [
        {'params': encoder.parameters(),},
        {'params': transformer.parameters(),},
        {'params': hnn.parameters(),},
        {'params': decoder.parameters(),},
    ]
    optimizer = torch.optim.SGD(optim_params, lr = 0.01, momentum=0.9)
    loss = torch.nn.MSELoss()

    # Instantiate Hamiltonian Generative Network
    hgn = HGN(encoder=encoder,
              transformer=transformer,
              hnn=hnn,
              decoder=decoder,
              integrator=hgn_integrator,
              loss=loss,
              optimizer=optimizer,
              device="cpu",
              dtype=torch.double,
              seq_len=rollouts.shape[1],
              channels=1)

    base_error = hgn.fit(rollouts)
    print(base_error)

    # print(torch.autograd.gradcheck(hgn.fit, rollouts))
    networks = [encoder, transformer, hnn, decoder]

    # Automatic gradients
    base_gradients = []
    print("Automatic gradients:")
    for network in networks:
        for param in network.parameters():
            base_gradients.append(param.grad.numpy())
    base_gradients = np.array(base_gradients).flatten()
    print(base_gradients)

    # Numeric gradients
    print("\nNumeric gradients:")
    num_grads = []
    for network in networks:
        net_grads = []
        for param in network.parameters():
            param_copy = param.data.clone()
            for indx, _ in np.ndenumerate(param_copy):
                param_copy[indx] += epsilon
                param.data = param_copy
                error = hgn.fit(rollouts)
                param_copy[indx] -= epsilon
                param.data = param_copy
                print(hgn.fit(rollouts))

                estimated_grad = (error - base_error)/epsilon
                net_grads.append(estimated_grad.detach().numpy())
        num_grads.append(np.array(net_grads))
    num_grads = np.array(num_grads)
    print(num_grads)

    print("\nRelative errors:")
    errors = 100*np.abs((base_gradients - num_grads)/num_grads)
    print(errors)

    # print("Average:", np.average(errors.flatten()))
    
