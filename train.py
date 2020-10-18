"""Script to train the Hamiltonian Generative Network
"""
import os

import time
import torch
import tqdm
import yaml

from environments.datasets import EnvironmentSampler
from environments.environments import visualize_rollout
from environments.environment_factory import EnvFactory
from hamiltonian_generative_network import HGN
from networks.inference_net import EncoderNet, TransformerNet
from networks.hamiltonian_net import HamiltonianNet
from networks.decoder_net import DecoderNet
from utilities.integrator import Integrator
from utilities.training_logger import TrainingLogger

params_file = "experiment_params/overfit_test.yaml"

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
        **params["networks"]["transformer"]).to(device)
    hnn = HamiltonianNet(**params["networks"]["hamiltonian"]).to(device)
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"],
        out_channels=params["rollout"]["n_channels"],
        **params["networks"]["decoder"]).to(device)

    # Define HGN integrator
    integrator = Integrator(delta_t=params["rollout"]["delta_time"],
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
        color=params["rollout"]["n_channels"] == 3,
        noise_std=params["dataset"]["noise_std"],
        radius_bound=params["dataset"]["radius_bound"],
        world_size=params["dataset"]["world_size"],
        seed=0)
    # Dataloader instance test, batch_mode disabled
    data_loader = torch.utils.data.DataLoader(trainDS,
                                              shuffle=False,
                                              batch_size=None)

    hgn.load(os.path.join(params["model_save_dir"], params["experiment_id"]))
    training_logger = TrainingLogger(loss_freq=100, rollout_freq=100)

    # Initialize tensorboard writer
    pbar = tqdm.tqdm(data_loader)
    for i, rollout_batch in enumerate(pbar):
        rollout_batch = rollout_batch.float().to(device)
        error, kld, prediction = hgn.fit(rollout_batch)
        training_logger.step(losses=(error, kld),
                             rollout_batch=rollout_batch,
                             prediction=prediction)
        msg = "Loss: %s, KL: %s" % (round(error, 4), round(kld, 4))
        pbar.set_description(msg)

    # import matplotlib.pyplot as plt
    # plt.plot(list(range(len(errors))), errors)
    # plt.plot(list(range(len(errors))), KLD_errors)
    # plt.show()
    # print("errors:\n", errors)
    hgn.save(os.path.join(params["model_save_dir"], params["experiment_id"]))

    # test_rollout = env.sample_random_rollouts(
    #     number_of_frames=params["rollout"]["seq_length"],
    #     delta_time=params["rollout"]["delta_time"],
    #     number_of_rollouts=1,
    #     img_size=params["dataset"]["img_size"],
    #     color=params["rollout"]["n_channels"] == 3,
    #     noise_std=params["dataset"]["noise_std"],
    #     radius_bound=params["dataset"]["radius_bound"],
    #     world_size=params["dataset"]["world_size"],
    #     seed=1)

    # test_rollout = test_rollout.transpose((0, 1, 4, 2, 3))
    # # visualize_rollout(test_rollout)
    # test_rollout = torch.tensor(test_rollout).float().to(device)
    # prediction = hgn.forward(test_rollout, n_steps=10)
    # for i in range(prediction.reconstructed_rollout.size()[1]):
    #     first_img = prediction.reconstructed_rollout[0, i].detach().numpy()
    #     cv2.imshow("img", first_img)
    #     cv2.waitKey(0)
    # prediction.visualize()
