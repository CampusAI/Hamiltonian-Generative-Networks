"""Script to train the Hamiltonian Generative Network
"""
import os

import torch
from tqdm import tqdm
import yaml

from envs.datasets import EnvironmentSampler
from envs.environments import visualize_rollout
from envs.environment_factory import EnvFactory
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

    # Set device
    device = "cuda:" + str(
        params["gpu_id"]) if torch.cuda.is_available() else "cpu"

    # Pick environment
    env = EnvFactory.get_environment(**params["environment"])

    # Instantiate networks
    encoder = EncoderNet(seq_len=params["rollout"]["seq_length"],
                         in_channels=params["rollout"]["n_channels"],
                         **params["networks"]["encoder"]).to(device)
    transformer = TransformerNet(
        in_channels=params["networks"]["encoder"]["out_channels"],
        **params["networks"]["transformer"])
    hnn = HamiltonianNet(**params["networks"]["hamiltonian"])
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"],
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

    # Dataloader
    dataset_len = params["optimization"]["epochs"] * params["optimization"][
        "batch_size"]
    trainDS = EnvironmentSampler(
        environment=env,
        dataset_len=dataset_len,
        number_of_frames=params["rollout"]["seq_length"],
        delta_time=params["rollout"]["delta_time"],
        number_of_rollouts=params["optimization"]["batch_size"],
        img_size=params["dataset"]["img_size"],
        noise_std=params["dataset"]["noise_std"],
        radius_bound=params["dataset"]["radius_bound"],
        world_size=params["dataset"]["world_size"],
        seed=None)
    # Dataloader instance test, batch_mode disabled
    data_loader = torch.utils.data.DataLoader(trainDS,
                                              shuffle=False,
                                              batch_size=None)
    errors = []
    for rollout_batch in tqdm(data_loader):
        rollout_batch = rollout_batch.float().to(device)
        error = hgn.fit(rollout_batch)
        errors.append(float(error))
    print("errors:\n", errors)

    test_rollout = env.sample_random_rollouts(
        number_of_frames=params["rollout"]["seq_length"],
        delta_time=params["rollout"]["delta_time"],
        number_of_rollouts=1,
        img_size=params["dataset"]["img_size"],
        noise_std=params["dataset"]["noise_std"],
        radius_bound=params["dataset"]["radius_bound"],
        world_size=params["dataset"]["world_size"],
        seed=1)
    
    # visualize_rollout(test_rollout)
    test_rollout = test_rollout.transpose((0, 1, 4, 2, 3))
    test_rollout = torch.tensor(test_rollout).float().to(device)
    print(test_rollout.size())
    prediction = hgn.forward(test_rollout, n_steps=10)
    prediction.visualize()

    hgn.save(os.path.join(params["model_save_dir"], params["experiment_id"]))